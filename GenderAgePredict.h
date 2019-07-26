#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>
#include <math.h>
#include <fstream>
#include "Retinaface.h"

using namespace mxnet::cpp;
using namespace std;
using namespace cv;
#define Amethod
class GenderAgeDetect
{

	Context *Ctx = nullptr;
	Symbol Sym_age;
	std::map<std::string, mxnet::cpp::NDArray> args_age;
	std::map<std::string, mxnet::cpp::NDArray> aux_age;

	Symbol Sym_gender;
	std::map<std::string, mxnet::cpp::NDArray> args_gender;
	std::map<std::string, mxnet::cpp::NDArray> aux_gender;
	int image_size = 64;

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

	NDArray GetDataBatch(Mat src, vector<Face> faces)
	{
		int len_img = image_size*image_size;
		float* data_img = new float[faces.size()*src.channels()*image_size*image_size];//batch_size channel h w
																					  
		for (size_t i = 0; i < faces.size(); i++)
		{
			int offset = src.channels()*i*len_img;

			Mat crop(src, preprocessRect(faces[i].boundingbox, src, 0.1));
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
	Rect preprocessRect(Rect rect, Mat src, double scale = 0.2)
	{
		Rect result(rect);
		if (rect.height > rect.width)
		{
			int padding = double(rect.height)*scale;
			result.y = std::max(0, result.y - padding); //拉高
			result.height += 2 * padding; //拉长
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
public:
	GenderAgeDetect(bool use_gpu, int imagesize = 64)
	{
		Ctx = use_gpu ? new Context(kGPU, 0) : new Context(kCPU, 0);
		image_size = imagesize;
	}
	~GenderAgeDetect()
	{
		delete Ctx;
	}

	void Loadmodel(String floder, String age_prefix,String gender_perfix)
	{
		Sym_age = Symbol::Load(floder + "/" + age_prefix + "-symbol.json");
		std::map<std::string, mxnet::cpp::NDArray> params_age;
		NDArray::Load(floder + "/" + age_prefix + "-0000.params", nullptr, &params_age);
		for (const auto &k : params_age)
		{
			if (k.first.substr(0, 4) == "aux:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				aux_age[name] = k.second.Copy(*Ctx);
			}
			if (k.first.substr(0, 4) == "arg:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				args_age[name] = k.second.Copy(*Ctx);
			}
		}

		Sym_gender = Symbol::Load(floder + "/" + gender_perfix + "-symbol.json");
		std::map<std::string, mxnet::cpp::NDArray> params_gender;
		NDArray::Load(floder + "/" + gender_perfix + "-0000.params", nullptr, &params_gender);
		for (const auto &k : params_gender)
		{
			if (k.first.substr(0, 4) == "aux:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				aux_gender[name] = k.second.Copy(*Ctx);
			}
			if (k.first.substr(0, 4) == "arg:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				args_gender[name] = k.second.Copy(*Ctx);
			}
		}

		// WaitAll is need when we copy data between GPU and the main memory
		mxnet::cpp::NDArray::WaitAll();
	}

	void detect(Mat src, vector<Face> faces, vector<mx_float>& agev, vector<string>& genderv, bool use_age = true, bool use_gender = true)
	{
		if (!use_age && !use_gender)
		{
			agev = vector<mx_float>(faces.size(), -1.0f);
			genderv = vector<string>(faces.size(), "unknown");
			return;
		}

		NDArray data = GetDataBatch(src, faces); //data2ndarray(*Ctx, data_img, 1, 3, img.rows, img.cols);

		vector<mx_float> stage = { 0,1,2 };
		vector<index_t> dim = { 1,3 };
		NDArray data_stage = GetNDArray(stage, dim, *Ctx);

		if (use_age)
		{
			args_age["data"] = data;
			args_age["stage_num0"] = data_stage;
			args_age["stage_num1"] = data_stage;
			args_age["stage_num2"] = data_stage;

			Executor *exec_age = Sym_age.SimpleBind(*Ctx, args_age, map<string, NDArray>(), map<string, OpReqType>(), aux_age);
			exec_age->Forward(false);
			std::vector<mx_float> age_data;
			exec_age->outputs[0].SyncCopyToCPU(&age_data, exec_age->outputs[0].Size());
			agev.insert(agev.end(), age_data.begin(), age_data.end());
			delete exec_age;
			exec_age = nullptr;
		}
		else
		{
			agev = vector<mx_float>(faces.size(), -1.0f);
		}

		if (use_gender)
		{
			args_gender["data"] = data;
			args_gender["stage_num0"] = data_stage;
			args_gender["stage_num1"] = data_stage;
			args_gender["stage_num2"] = data_stage;

			Executor *exec_gender = Sym_age.SimpleBind(*Ctx, args_gender, map<string, NDArray>(), map<string, OpReqType>(), aux_gender);
			exec_gender->Forward(false);
			std::vector<mx_float> gender_data;
			exec_gender->outputs[0].SyncCopyToCPU(&gender_data, exec_gender->outputs[0].Size());
			for (size_t i = 0; i < gender_data.size(); i++)
			{
				if (gender_data[i] > 0.5)
				{
					genderv.push_back("male");
					continue;
				}
				genderv.push_back("female");
			}
			delete exec_gender;
			exec_gender = nullptr;
		}
		else
		{

			genderv = vector<string>(faces.size(), "unknown");
		}



	}

};
