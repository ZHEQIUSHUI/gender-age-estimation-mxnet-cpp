#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>
#include <math.h>
#include <fstream>
#include <numeric>
#include "Retinaface.h"
#include "FacePreprocess.h"

using namespace mxnet::cpp;
using namespace std;
using namespace cv;

#define Bmethod
class GenderAgeDetect
{

	Context *Ctx = nullptr;
	Symbol Sym;
	std::map<std::string, mxnet::cpp::NDArray> args;
	std::map<std::string, mxnet::cpp::NDArray> aux;

	int image_size = 112;

	Point2f target[5] = { Point2f(30.2946, 51.6963),
		Point2f(65.5318, 51.5014),
		Point2f(48.0252, 71.7366),
		Point2f(33.5493, 92.3655),
		Point2f(62.7299, 92.2041) };
	Mat _dst = Mat(5, 2, CV_32F);

private:
	mxnet::cpp::NDArray data2ndarray(mxnet::cpp::Context ctx, float * data, int batch_size, int num_channels, int height, int width)
	{
		mxnet::cpp::NDArray ret(mxnet::cpp::Shape(batch_size, num_channels, height, width), ctx, false);

		ret.SyncCopyFromCPU(data, batch_size * num_channels * height * width);

		ret.WaitToRead();  //mxnet::cpp::NDArray::WaitAll();

		return ret;
	}

	NDArray GetNDArray(vector<mx_float> data, vector<index_t> data_shape, Context ctx)
	{
		mxnet::cpp::NDArray ret(mxnet::cpp::Shape(data_shape), ctx, false);

		ret.SyncCopyFromCPU(data);

		ret.WaitToRead();  //mxnet::cpp::NDArray::WaitAll();

		return ret;
	}

	NDArray GetNDArray(mx_float* data, vector<index_t> data_shape, Context ctx)
	{
		int size = 1;
		for (size_t i = 0; i < data_shape.size(); i++)
		{
			size *= data_shape[i];
		}
		mxnet::cpp::NDArray ret(mxnet::cpp::Shape(data_shape), ctx, false);

		ret.SyncCopyFromCPU(data, size);

		ret.WaitToRead();  //mxnet::cpp::NDArray::WaitAll();

		return ret;
	}

	Rect preprocessRect(Rect rect, Mat src, double scale = 0.2)
	{
		Rect result(rect);
		if (rect.height > rect.width)
		{
			int padding = double(rect.height)*scale;
			result.y = std::max(0, result.y - padding); //拉高
			result.height += 2*padding; //拉长
			int offset = (result.height - result.width) / 2; //长宽差/2
			result.x = std::max(0, result.x - offset); //向左
			result.width += 2 * offset; //拉宽
		}
		else
		{
			int padding = double(rect.width)*scale;
			result.x = std::max(0, result.x - padding); //向左
			result.width += 2 * padding;  //拉宽
			int offset = (result.width - result.height) / 2; //长宽差/2
			result.y = std::max(0, result.y - offset); //向上
			result.height += 2 * offset; //拉长
		}
		result.width = std::min(src.cols - result.x, result.width);
		result.height = std::min(src.rows - result.y, result.height);
		return result;
	}

	Mat alignFace(Mat src, Face face, Mat dst)
	{
		Mat _src(5, 2, CV_32F);
		for (size_t i = 0; i < _src.rows; i++)
		{
			_src.at<float>(i, 0) = face.landmarks[i].x;
			_src.at<float>(i, 1) = face.landmarks[i].y;
		}
		Mat m = FacePreprocess::similarTransform(_src, dst);
		Mat _m = m.rowRange(0, 2);
		Mat warp;
		warpAffine(src, warp, _m, Size(112, 112));
		return warp;
	}

	NDArray GetDataBatch(Mat src, vector<Face> faces)
	{
		int len_img = image_size*image_size;
		float* data_img = new float[faces.size()*src.channels()*image_size*image_size];//batch_size channel h w
																					 
		for (size_t i = 0; i < faces.size(); i++)
		{
			int offset = i*src.channels()*len_img;

			Mat crop = alignFace(src, faces[i], _dst);
			Mat img;
			resize(crop, img, Size(image_size, image_size));
			img.convertTo(img, CV_32FC3);
			Mat bgr[3];
			split(img, bgr);
			Mat b_img = bgr[0];
			Mat g_img = bgr[1];
			Mat r_img = bgr[2];

			memcpy(data_img + offset, r_img.data, len_img * sizeof(*data_img));
			memcpy(data_img + offset + len_img, g_img.data, len_img * sizeof(*data_img));
			memcpy(data_img + offset + len_img + len_img, b_img.data, len_img * sizeof(*data_img));

			img.release();
			b_img.release();
			g_img.release();
			r_img.release();
		}
		vector<index_t> shape = { (uint)faces.size(),(uint)src.channels(),(uint)image_size,(uint)image_size };
		NDArray db = GetNDArray(data_img, shape, *Ctx);
		delete[] data_img;
		data_img = nullptr;
		return db;
	}

public:
	GenderAgeDetect(bool use_gpu)
	{
		Ctx = use_gpu ? new Context(kGPU, 0) : new Context(kCPU, 0);

		for (size_t i = 0; i < _dst.rows; i++)
		{
			_dst.at<float>(i, 0) = target[i].x + 8.0;
			_dst.at<float>(i, 1) = target[i].y;
		}
	}
	~GenderAgeDetect()
	{
		delete Ctx;
	}

	void Loadmodel(String floder, String prefix)
	{
		Sym = Symbol::Load(floder + "/" + prefix + "-symbol.json");
		std::map<std::string, mxnet::cpp::NDArray> params_age;
		NDArray::Load(floder + "/" + prefix + "-0000.params", nullptr, &params_age);
		for (const auto &k : params_age)
		{
			if (k.first.substr(0, 4) == "aux:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				aux[name] = k.second.Copy(*Ctx);
			}
			if (k.first.substr(0, 4) == "arg:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				args[name] = k.second.Copy(*Ctx);
			}
		}


		// WaitAll is need when we copy data between GPU and the main memory
		mxnet::cpp::NDArray::WaitAll();
	}

	

	void detect(Mat src, vector<Face> faces, vector<mx_float>& agev, vector<string>& genderv, bool use_age = true, bool use_gender = true)
	{
		NDArray data = GetDataBatch(src, faces); //data2ndarray(*Ctx, data_img, 1, 3, img.rows, img.cols);

		args["data"] = data;

		Executor *exec = Sym.SimpleBind(*Ctx, args, map<string, NDArray>(), map<string, OpReqType>(), aux);
		exec->Forward(false);
		std::vector<mx_float> output_data;
		exec->outputs[0].SyncCopyToCPU(&output_data, exec->outputs[0].Size());

		for (size_t i = 0; i < faces.size(); i++)
		{
			int offset = i * 202;
			mx_float a = output_data[0 + offset];
			mx_float b = output_data[1 + offset];
			genderv.push_back(b > a ? "male" : "female");

			vector<mx_float> ages_temp(output_data.begin() + offset + 2, output_data.begin() + offset + 202);
			vector<int> ages;
			for (size_t j = 0; j < 100; j++)
			{
				ages.push_back(ages_temp[2 * j] > ages_temp[2 * j + 1] ? 0 : 1);
			}
			int age = accumulate(ages.begin(), ages.end(), 0);
			agev.push_back(age);
		}

		delete exec;
		exec = nullptr;
	}

};