/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Itseez Inc, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "dpm_cascade.hpp"
#include "dpm_nms.hpp"

#include <limits>
#include <fstream>
#include <iostream>
#include <stdio.h>

using namespace std;

namespace cv
{
namespace dpm
{

const static int NUM_DIRECTION = 8;
const static int dy[] = {-1,-1,-1,0,0,1,1,1};
const static int dx[] = {-1,0,1,-1,1,-1,0,1};

void DPMCascade::loadCascadeModel(const string &modelPath)
{
	// load cascade model from xml
	bool is_success = model.deserialize(modelPath);

	if (!is_success)
	{
		string errorMessage = format("Unable to parse the model: %s", modelPath.c_str());
		CV_Error(CV_StsBadArg, errorMessage);
	}

	model.initModel();

	FeatureGPUParams paramsG;
	paramsG.pad_x = model.maxSizeX;
	paramsG.pad_y = model.maxSizeY;
	paramsG.interval = model.interval;
	paramsG.sbin = model.sBin;

	fg.initialize(paramsG);
	fg.setPcaCoeff(model.pcaCoeff);

	PyramidParameter paramsC;
	paramsC.padx = model.maxSizeX;
	paramsC.pady = model.maxSizeY;
	paramsC.interval = model.interval;
	paramsC.binSize = model.sBin;

	feature = Feature(paramsC);
}

void DPMCascade::initDPMCascade()
{
	// compute the size of temporary storage needed by cascade
	int nlevels = (int) pyramid.size();
	int numPartFilters = model.getNumPartFilters();
	int numDefParams = model.getNumDefParams();
	featDimsProd.resize(nlevels);
	tempStorageSize = 0;

	for (int i = 0; i < nlevels; i++)
	{
		int w = pyramid[i].cols/feature.dimHOG;
		int h = pyramid[i].rows;
		featDimsProd[i] = w*h;
		tempStorageSize += w*h;
	}

	tempStorageSize *= numPartFilters;

	convValues.resize(tempStorageSize);
	pcaConvValues.resize(tempStorageSize);
	dtValues.resize(tempStorageSize);
	pcaDtValues.resize(tempStorageSize);

	fill(convValues.begin(), convValues.end(), -numeric_limits<float>::infinity());
	fill(pcaConvValues.begin(), pcaConvValues.end(), -numeric_limits<float>::infinity());
	fill(dtValues.begin(), dtValues.end(), -numeric_limits<float>::infinity());
	fill(pcaDtValues.begin(), pcaDtValues.end(), -numeric_limits<float>::infinity());

	// each pyramid (convolution and distance transform) is stored
	// in a 1D array. Since pyramid levels have different sizes,
	// we build an array of offset values in order to index by
	// level. The last offset is the total length of the pyramid
	// storage array.
	convLevelOffset.resize(nlevels + 1);
	dtLevelOffset.resize(nlevels + 1);
	convLevelOffset[0] = 0;
	dtLevelOffset[0] = 0;

	for (int i = 1; i < nlevels + 1; i++)
	{
		convLevelOffset[i] = convLevelOffset[i-1] + numPartFilters*featDimsProd[i-1];
		dtLevelOffset[i] = dtLevelOffset[i-1] + numDefParams*featDimsProd[i-1];
	}

	// cache of precomputed deformation costs
	defCostCacheX.resize(numDefParams);
	defCostCacheY.resize(numDefParams);

	for (int i = 0; i < numDefParams; i++)
	{
		vector< float > def = model.defs[i];
		CV_Assert((int) def.size() >= 4);

		defCostCacheX[i].resize(2*halfWindowSize + 1);
		defCostCacheY[i].resize(2*halfWindowSize + 1);

		for (int j = 0; j < 2*halfWindowSize + 1; j++)
		{
			int delta = j - halfWindowSize;
			int deltaSquare = delta*delta;
			defCostCacheX[i][j] = -def[0]*deltaSquare - def[1]*delta;
			defCostCacheY[i][j] = -def[2]*deltaSquare - def[3]*delta;
		}
	}

	dtArgmaxX.resize(dtLevelOffset[nlevels]);
	pcaDtArgmaxX.resize(dtLevelOffset[nlevels]);
	dtArgmaxY.resize(dtLevelOffset[nlevels]);
	pcaDtArgmaxY.resize(dtLevelOffset[nlevels]);
}

vector< vector<float> > DPMCascade::detect(Mat &image)
{
	if (image.channels() == 1)
		cvtColor(image, image, COLOR_GRAY2BGR);

	if (image.depth() != CV_32F)
		image.convertTo(image, CV_32FC3);

	// compute features
	computeFeatures(image);

	// pre-allocate storage
	initDPMCascade();

	// cascade process
	vector< vector<float> > detections;
	process(detections);

	// non-maximum suppression
	NonMaximumSuppression nms;
	nms.process(detections, 0.5);

	return detections;
}

void DPMCascade::computeFeatures(const Mat &im)
{
	feature.computeScales(im);

	fg.loadImage(im);

	// compute pyramid
	fg.computeHistPyramid();

	fg.downloadFeature(pyramid);

	// compute projected pyramid
	// feature.projectFeaturePyramid(model.pcaCoeff, pyramid, pcaPyramid);
	fg.projectFeaturePyramid();
	fg.downloadPcaFeature(pcaPyramid);
}

void DPMCascade::computeLocationScores(vector< vector< float > >  &locationScores)
{
	vector< vector < float > > locationWeight = model.locationWeight;
	CV_Assert((int)locationWeight.size() == model.numComponents);

	Mat locationFeature;
	int nlevels = (int) pyramid.size();
	feature.computeLocationFeatures(nlevels, locationFeature);

	locationScores.resize(model.numComponents);

	for (int comp = 0; comp < model.numComponents; comp++)
	{
		locationScores[comp].resize(locationFeature.cols);

		for (int level = 0; level < locationFeature.cols; level++)
		{
			float val = 0;
			for (int k = 0; k < locationFeature.rows; k++)
				val += locationWeight[comp][k]*
				locationFeature.at<float>(k, level);

			locationScores[comp][level] = val;
		}
	}
}

void DPMCascade::computeRootPCAScores(vector< vector< Mat > > &rootScores)
{
	PyramidParameter params = feature.getPyramidParameters();
	rootScores.resize(model.numComponents);
	int nlevels = (int) pyramid.size();
	int interval = params.interval;

	for (int comp = 0; comp < model.numComponents; comp++)
	{
		rootScores[comp].resize(nlevels);
#ifdef HAVE_TBB // parallel computing
		ParalComputeRootPCAScores paralTask(pcaPyramid, model.rootPCAFilters[comp],
				model.pcaDim, rootScores[comp]);
		parallel_for_(Range(interval, nlevels), paralTask);
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int level = interval; level < nlevels; level++)
		{
			Mat feat = pcaPyramid[level];
			Mat filter = model.rootPCAFilters[comp];

			// compute size of output
			int height = feat.rows - filter.rows + 1;
			int width = (feat.cols - filter.cols) / model.pcaDim + 1;

			if (height < 1 || width < 1)
				CV_Error(CV_StsBadArg,
						"Invalid input, filter size should be smaller than feature size.");

			Mat result = Mat::zeros(Size(width, height), CV_32F);
			convolutionEngine.convolve(feat, filter, model.pcaDim, result);
			rootScores[comp][level] = result;
		}
#endif
	}
}

#ifdef HAVE_TBB
ParalComputeRootPCAScores::ParalComputeRootPCAScores(
		const vector< Mat > &pcaPyrad,
		const Mat &f,
		int dim,
		vector< Mat > &sc):
    						pcaPyramid(pcaPyrad),
    						filter(f),
    						pcaDim(dim),
    						scores(sc)
{
}

void ParalComputeRootPCAScores::operator() (const Range &range) const
{
	for (int level = range.start; level != range.end; level++)
	{
		Mat feat = pcaPyramid[level];

		// compute size of output
		int height = feat.rows - filter.rows + 1;
		int width = (feat.cols - filter.cols) / pcaDim + 1;

		Mat result = Mat::zeros(Size(width, height), CV_32F);
		// convolution engine
		ConvolutionEngine convEngine;
		convEngine.convolve(feat, filter, pcaDim, result);
		scores[level] = result;
	}
}
#endif

void DPMCascade::process( vector< vector<float> > &dets)
{
	PyramidParameter params = feature.getPyramidParameters();
	int interval = params.interval;
	int padx = params.padx;
	int pady = params.pady;
	vector<float> scales = params.scales;

	int nlevels = (int)pyramid.size() - interval;
	CV_Assert(nlevels > 0);

	// compute location scores
	vector< vector< float > > locationScores;
	computeLocationScores(locationScores);

	// compute root PCA scores
	vector< vector< Mat > > rootPCAScores;
	computeRootPCAScores(rootPCAScores);

	// process each model component and pyramid level
	for (int comp = 0; comp < model.numComponents; comp++)
	{
		for (int plevel = 0; plevel < nlevels; plevel++)
		{
			// root filter pyramid level
			int rlevel = plevel + interval;
			float bias = model.bias[comp] + locationScores[comp][rlevel];
			int numstages = 2*model.numParts[comp] + 2;

			// get the scores of the first PCA filter
			Mat rtscore = rootPCAScores[comp][rlevel];

			// keep track of the PCA scores for each PCA filter
			vector< vector< vector< float > > > pcaScore(rtscore.cols * rtscore.rows);
			for (int i = 0; i < pcaScore.size(); i++)
			{
				pcaScore[i].resize(model.numComponents);
				for (int comp = 0; comp < model.numComponents; comp++)
					pcaScore[i][comp].resize(model.numParts[comp]+1);
			}

			/* vector<Mat> stageScore;
			for (int i = 0; i < numstages; i++)
				stageScore.push_back(Mat(rtscore.size(), CV_32F, Scalar(-1))); */

			Mat currScore(rtscore.size(), CV_32F, Scalar(0));
			Mat mask(rtscore.size(), CV_8U, Scalar(1));

			// process each location in the current pyramid level
			// TODO: move <stage> to the outermost loop, save scores to a Mat named <stageScore>
			int stage = 0;
			// cascade stage 1 through 2*numparts + 2
			for (; stage <= numstages; stage++)
			{
				for (int ry = (int)ceil(pady/2.0); ry < rtscore.rows - (int)ceil(pady/2.0); ry++)
				{
					for (int rx = (int)ceil(padx/2.0); rx < rtscore.cols - (int)ceil(padx/2.0); rx++)
					{
						if (stage == 0)
						{
							// get stage 0 score
							float score = rtscore.at<float>(ry, rx) + bias;
							// record PCA score
							pcaScore[ry * rtscore.rows + rx][comp][0] = score - bias;
							currScore.at<float>(ry, rx) = score;
							// stageScore[stage].at<float>(ry, rx) = score;
						}

						if (stage > 0 && stage < numstages)
						{
							if (mask.at<uchar>(ry, rx) == 0)
								continue;

							float score = currScore.at<float>(ry, rx);
							float t = model.prunThreshold[comp][2*stage-1];
							// check for hypothesis pruning


							// TODO: add semiNegativeThreshold for semi-negative pruning
							// semi-negative pruning
							// float semi_negative_thres = model.semiNegativeThreshold[comp][2 * stage - 1];
							if (score < t - 0.6)
							{
								mask.at<uchar>(ry, rx) = 0;
								for (int k = 0; k < NUM_DIRECTION; k++)
									mask.at<uchar>(ry + dy[k], rx + dx[k]) = 0;
								continue;
							}

							if (score < t)
							{
								mask.at<uchar>(ry, rx) = 0;
								continue;
							}

							// pca == 1 if place filters
							// pca == 0 if place non-pca filters
							bool isPCA = (stage < model.numParts[comp] + 1 ? true : false);
							// get the part index
							// root parts have index -1, none-root part are indexed 0:numParts-1
							int part = model.partOrder[comp][stage] - 1;// partOrder

							auto& localPcaScore = pcaScore[ry * rtscore.rows + rx];

							if (part == -1)
							{
								// calculate the root non-pca score
								// and replace the PCA score
								float rscore = 0.0;
								if (isPCA)
								{
									rscore = convolutionEngine.convolve(pcaPyramid[rlevel],
											model.rootPCAFilters[comp],
											model.pcaDim, rx, ry);
								}
								else
								{
									rscore = convolutionEngine.convolve(pyramid[rlevel],
											model.rootFilters[comp],
											model.numFeatures, rx, ry);
								}
								score += rscore - localPcaScore[comp][0];
							}
							else
							{
								// place a non-root filter
								int pId = model.pFind[comp][part];
								int px = 2*rx + (int)model.anchors[pId][0];
								int py = 2*ry + (int)model.anchors[pId][1];

								// look up the filter and deformation model
								float defThreshold =
										model.prunThreshold[comp][2*stage] - score;
								// TODO: commented this for data retrieve, Yilong Li 5/11

								double ps = computePartScore(plevel, pId, px, py,
										isPCA, defThreshold);

								if (isPCA)
								{
									// record PCA filter score
									localPcaScore[comp][part+1] = ps;
									// update the hypothesis score
									score += ps;
								}
								else
								{
									// update the hypothesis score by replacing
									// the PCA score
									score += ps - localPcaScore[comp][part+1];
								} // isPCA == false
							} // part != -1

							currScore.at<float>(ry, rx) = score;

							// for debug purpose
							// stageScore[stage].at<float>(ry, rx) = score;
						}

						// check if the hypothesis passed all stages with a
						// final score over the global threshold
						if (stage == numstages && currScore.at<float>(ry, rx) >= model.scoreThresh)
						{
							float score = currScore.at<float>(ry, rx);

							cerr << "DETECTION!" << rlevel << " " <<  ry <<  " " << rx << " " << score << endl;

							vector<float> coords;
							// compute and record image coordinates of the detection window
							float scale = model.sBin/scales[rlevel];
							float x1 = (rx-padx)*scale;
							float y1 = (ry-pady)*scale;
							float x2 = x1 + model.rootFilterDims[comp].width*scale - 1;
							float y2 = y1 + model.rootFilterDims[comp].height*scale - 1;

							coords.push_back(x1);
							coords.push_back(y1);
							coords.push_back(x2);
							coords.push_back(y2);

							// compute and record image coordinates of the part filters
							scale = model.sBin/scales[plevel];
							int featWidth = pyramid[plevel].cols/feature.dimHOG;
							for (int p = 0; p < model.numParts[comp]; p++)
							{
								int pId = model.pFind[comp][p];
								int probx = 2*rx + (int)model.anchors[pId][0];
								int proby = 2*ry + (int)model.anchors[pId][1];
								int offset = dtLevelOffset[plevel] +
										pId*featDimsProd[plevel] +
										(proby - pady)*featWidth +
										probx - padx;
								int px = dtArgmaxX[offset] + padx;
								int py = dtArgmaxY[offset] + pady;
								x1 = (px - 2*padx)*scale;
								y1 = (py - 2*pady)*scale;
								x2 = x1 + model.partFilterDims[p].width*scale - 1;
								y2 = y1 + model.partFilterDims[p].height*scale - 1;
								coords.push_back(x1);
								coords.push_back(y1);
								coords.push_back(x2);
								coords.push_back(y2);
							}

							// record component number and score
							coords.push_back(comp + 1);
							coords.push_back(score);

							dets.push_back(coords);
						}
					} // rx
				} // ry
			}//stage

			/*
			cout << "stages = []" << endl;
			cout << "plevel = " << plevel << endl <<
					"compid  = " << comp << endl << endl;

			for (int i = 0; i < numstages; i ++)
			{
				cout << "tmp = ";
				cout << format(stageScore[i], "numpy") << endl;
				cout << "stages.append(tmp)" << endl;
			} */
		} // for each pyramid level
	} // for each component
}

float DPMCascade::computePartScore(int plevel, int pId, int px, int py, bool isPCA, float defThreshold)
{
	// remove virtual padding
	PyramidParameter params = feature.getPyramidParameters();
	px -= params.padx;
	py -= params.pady;

	// check if already computed
	int levelOffset = dtLevelOffset[plevel];
	int locationOffset = pId*featDimsProd[plevel]
	                                      + py*pyramid[plevel].cols/feature.dimHOG
	                                      + px;
	int dtBaseOffset = levelOffset + locationOffset;

	float val;

	if (isPCA)
		val = pcaDtValues[dtBaseOffset];
	else
		val = dtValues[dtBaseOffset];

	if (val > -numeric_limits<float>::infinity())
		return val;

	// Nope, define the bounds of the convolution and
	// distance transform region
	int xstart = px - halfWindowSize;
	xstart = (xstart < 0 ? 0 : xstart);
	int xend = px + halfWindowSize;

	int ystart = py - halfWindowSize;
	ystart = (ystart < 0 ? 0 : ystart);
	int yend = py + halfWindowSize;

	int featWidth = pyramid[plevel].cols/feature.dimHOG;
	int featHeight = pyramid[plevel].rows;
	int filterWidth = model.partFilters[pId].cols/feature.dimHOG;
	int filterHeight = model.partFilters[pId].rows;

	xend = (filterWidth + xend > featWidth)
        						? featWidth - filterWidth
        								: xend;
	yend = (filterHeight + yend > featHeight)
        						? featHeight - filterHeight
        								: yend;

	// do convolution and distance transform in region
	// [xstar, xend, ystart, yend]
	levelOffset = convLevelOffset[plevel];
	locationOffset = pId*featDimsProd[plevel];
	int convBaseOffset = levelOffset + locationOffset;

	for (int y = ystart; y <= yend; y++)
	{
		int loc = convBaseOffset + y*featWidth + xstart - 1;
		for (int x = xstart; x <= xend; x++)
		{
			loc++;
			// skip if already computed
			if (isPCA)
			{
				if (pcaConvValues[loc] > -numeric_limits<float>::infinity())
					continue;
			}
			else if(convValues[loc] > -numeric_limits<float>::infinity())
				continue;

			// check for deformation pruning
			float defCost = defCostCacheX[pId][px - x + halfWindowSize]
			                                   + defCostCacheY[pId][py - y + halfWindowSize];

			if (defCost < defThreshold)
				continue;

			if (isPCA)
			{
				pcaConvValues[loc] = convolutionEngine.convolve
						(pcaPyramid[plevel], model.partPCAFilters[pId],
								model.pcaDim, x, y);
			}
			else
			{
				convValues[loc] = convolutionEngine.convolve
						(pyramid[plevel], model.partFilters[pId],
								model.numFeatures, x, y);
			}
		} // y
	} // x

	// do distance transform over the region.
	// the region is small enought that brut force DT
	// is the fastest method
	float max = -numeric_limits<float>::infinity();
	int xargmax = 0;
	int yargmax = 0;

	for (int y = ystart; y <= yend; y++)
	{
		int loc = convBaseOffset + y*featWidth + xstart - 1;
		for (int x = xstart; x <= xend; x++)
		{
			loc++;
			float v;

			if (isPCA)
				v = pcaConvValues[loc];
			else
				v = convValues[loc];

			v += defCostCacheX[pId][px - x + halfWindowSize]
			                        + defCostCacheY[pId][py - y + halfWindowSize];

			if (v > max)
			{
				max = v;
				xargmax = x;
				yargmax = y;
			} // if v
		} // for x
	} // for y

	// record max and argmax for DT
	if (isPCA)
	{
		pcaDtArgmaxX[dtBaseOffset] = xargmax;
		pcaDtArgmaxY[dtBaseOffset] = yargmax;
		pcaDtValues[dtBaseOffset] = max;
	}
	else
	{
		dtArgmaxX[dtBaseOffset] = xargmax;
		dtArgmaxY[dtBaseOffset] = yargmax;
		dtValues[dtBaseOffset] = max;
	}

	return max;
}

} // namespace dpm
} // namespace cv
