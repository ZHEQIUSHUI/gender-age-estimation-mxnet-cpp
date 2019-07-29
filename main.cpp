#pragma comment(lib,"opencv_world410.lib")
#pragma comment(lib,"libmxnet.lib")


#include "Retinaface.h"
#include "GenderAgePredict_othermethod.h"
#include "FacePreprocess.h"


void DrawAgeGenderScore(Mat& src, vector<Face> faces, vector<string> genderv, vector<mx_float> agev,bool use_landmarks)
{
	//cout << "find " << faces.size() << " faces fps " << fps / times << endl;
	for (size_t i = 0; i < faces.size(); i++)
	{
		rectangle(src, faces[i].boundingbox, Scalar(0, 255, 0), 2);

		putText(src, to_string(faces[i].score), Point(faces[i].boundingbox.x + 2, faces[i].boundingbox.y + 20), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 1);

		if (use_landmarks)
		{
			putText(src, genderv[i], Point(faces[i].boundingbox.x + 2, faces[i].boundingbox.y + 40), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 1);
			putText(src, "Age:" + to_string((int)agev[i]), Point(faces[i].boundingbox.x + 2, faces[i].boundingbox.y + 60), FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 1);

			for (size_t j = 0; j < faces[i].landmarks.size(); j++)
			{
				circle(src, faces[i].landmarks[j], 1, Scalar(255, 0, 0), 2);
			}
		}
		
	}
}


int main(int argc,char* args[])
{
	if (argc <= 1)
	{
		cout << "\\\> retinaface.exe " << endl;
		cout << "argv:" << endl;
		cout << "             -v/-i video/image path           (type:string,-v videopath(0?webcam:videofile),-i imagepath,default=0)" << endl;
		cout << "             -s    scale                      (type:float,default=1.0)" << endl;
		cout << "             -t    score_threshold            (type:float,default=0.8)" << endl;
		cout << "             -l    use_landmarks              (type:int,default=1)" << endl;
		cout << "             -g    use_gpu                    (type:int,default=1)" << endl;
		cout << "             -gender    use_gender            (type:int,default=1)" << endl;
		cout << "             -age       use_age               (type:int,default=1)" << endl;

		cout << "Example:" << endl << endl;
		cout << ">>>>>>>>>>>>> detect in image <<<<<<<<<<<<<" << endl;
		cout << "\\\> .\\retinaface.exe -i .\\people.jpeg -s 1.0 -t 0.8 -l 1 -g 1 -age 1 -gender 1" << endl << endl;
		cout << ">>>>>>>>>>>>> detect in video <<<<<<<<<<<<<" << endl;
		cout << "\\\> .\\retinaface.exe -v .\\multiple_faces.avi -s 0.5 -t 0.8 -l 0 -g 1 -age 1 -gender 1" << endl << endl;
		cout << ">>>>>>>>>>>>> detect in webcam <<<<<<<<<<<<<" << endl;
		cout << "\\\> .\\retinaface.exe -v 0 -s 0.5 -t 0.8 -l 1 -g 1 -age 1 -gender 1" << endl << endl << endl;
		return 0;
	}
	//dlog(argc);
	bool isvideo = true;
	bool use_lankmarks = 1;
	bool use_gpu = 1;
	bool use_gender = 1;
	bool use_age = 1;
	string path = "0";
	mx_float scale = 1.0;
	mx_float score = 0.8;
	for (size_t i = 1; i < argc; i++)
	{
		if (string(args[i]) == "-v")
		{
			isvideo = true;
			path = args[i + 1];
		}
		else if (string(args[i]) == "-i")
		{
			isvideo = false;
			path = args[i + 1];
		}
		else if (string(args[i]) == "-s")
		{
			scale = atof(args[i + 1]);
		}
		else if (string(args[i]) == "-t")
		{
			score = atof(args[i + 1]);
		}
		else if (string(args[i]) == "-l")
		{
			use_lankmarks = atoi(args[i + 1]);
		}
		else if (string(args[i]) == "-g")
		{
			use_gpu = atoi(args[i + 1]);
		}
		else if (string(args[i]) == "-gender")
		{
			use_gender = atoi(args[i + 1]);
		}
		else if (string(args[i]) == "-age")
		{
			use_age = atoi(args[i + 1]);
		}
	}
	cout << "scale :" << scale << endl;
	cout << "score :" << score << endl;
	cout << "use_landmarks? :" << use_lankmarks << endl;
	cout << "use_gpu? :" << use_gpu << endl;
	cout << "use_gender? :" << use_gender << endl;
	cout << "use_age? :" << use_age << endl;
	RetinaFace m_facedetector(use_gpu);
	m_facedetector.Loadmodel("E:/PyCode/insightface/models", "mnet.25");
	m_facedetector.use_landmarks = use_lankmarks;

	GenderAgeDetect m_genderage_detector(use_gpu);

#ifdef Amethod
	m_genderage_detector.Loadmodel("models", "age", "gender");
#elif defined Bmethod
	m_genderage_detector.Loadmodel("models", "model");
#endif // Amethod

	

	Mat src;
	if (isvideo)
	{
		cout << "video_path :" << path << endl;
		VideoCapture *video = nullptr;
		if (path == "0")
		{
			video = new VideoCapture(0);
		}
		else
		{
			video = new VideoCapture(path);
		}
		double totalframecount = video->get(CAP_PROP_FRAME_COUNT);
		time_t start, end;
		start = clock();
		double fps = 0;
		while (true)
		{
			*video >> src;
			if (!src.data)
			{
				cout << "video is end" << endl;
				if (path == "0")
				{
					break;
				}
				video->set(CAP_PROP_POS_FRAMES, 0);
				fps = 0;
				start = clock();
				continue;
			}
			vector<Face> faces = m_facedetector.detect(src, score, vector<mx_float> {scale});
			

			vector<mx_float> agev;
			vector<string> genderv;
			if (faces.size()&&use_lankmarks)
			{
				m_genderage_detector.detect(src, faces, agev, genderv, use_age, use_gender);
			}
			DrawAgeGenderScore(src, faces, genderv, agev, use_lankmarks);
			end = clock();
			double times = double(end - start) / 1000.0;
			fps++;
			printf("\r");
			printf("find %d faces fps %f ,remainder %.0f playpercent %.2f%%", faces.size(), fps / times, totalframecount - fps, (fps * 100) / totalframecount);

			imshow(path, src);
			if (waitKey(1) == 27)
			{
				break;
			}
		}
		video->release();
		delete video;
		video = nullptr;
		
	}
	else
	{
		cout << "image_path :" << path << endl;
		src = imread(path);
		vector<Face> faces = m_facedetector.detect(src, score, vector<mx_float> {scale});
		cout << "find " << faces.size() << " faces"<< endl;
		vector<mx_float> agev;
		vector<string> genderv;
		if (faces.size()&&use_lankmarks)
		{
			m_genderage_detector.detect(src, faces, agev, genderv, use_age, use_gender);
		}
		DrawAgeGenderScore(src, faces, genderv, agev,use_lankmarks);
		imshow(path, src);
	}
	waitKey(0);

	//RetinaFace m_facedetector(1);
	//m_facedetector.Loadmodel("E:/PyCode/insightface/models", "mnet.25");
	//m_facedetector.use_landmarks = 1;
	//VideoCapture video(0);
	//Mat src;
	//while (true)
	//{
	//	video >> src;
	//	vector<Face> faces = m_facedetector.detect(src, 0.5, vector<mx_float> {1.0});
	//	Point2f target[5] = {	   Point2f(30.2946, 51.6963),
	//							   Point2f(65.5318, 51.5014),
	//							   Point2f(48.0252, 71.7366),
	//						       Point2f(33.5493, 92.3655),
	//							   Point2f(62.7299, 92.2041) };

	//	Mat _dst(5, 2, CV_32F);
	//	for (size_t i = 0; i < _dst.rows; i++)
	//	{
	//		_dst.at<float>(i, 0) = target[i].x + 8.0;
	//		_dst.at<float>(i, 1) = target[i].y;
	//	}
	//	Mat warp = alignFace(src, faces[0], _dst);
	//	/*Mat _src(5, 2, CV_32F);
	//	for (size_t i = 0; i < _src.rows; i++)
	//	{
	//		_src.at<float>(i, 0) = faces[0].landmarks[i].x;
	//		_src.at<float>(i, 1) = faces[0].landmarks[i].y;
	//	}
	//	Mat m = FacePreprocess::similarTransform(_src, _dst);
	//	Mat _m = m.rowRange(0, 2);
	//
	//	Mat warp;
	//	warpAffine(src, warp, _m, Size(112, 112));*/
	//	
	//	imshow("align", warp);
	//	rectangle(src,faces[0].boundingbox,Scalar(0,0,255));
	//	imshow("src", src);
	//	if (waitKey(1)==27)
	//	{
	//		break;
	//	}
	//}




	/*GenderAgeDetect m_genderage_detector(false);
	m_genderage_detector.Loadmodel("models", "model");

	Mat src = imread("E:\\face.png");
	vector<Face> faces = { Face{Rect(0,0,src.cols,src.rows),1.0f,vector<Point2f>{}} };
	vector<int> agev;
	vector<string> genderv;
	m_genderage_detector.detect(src, faces, agev, genderv);
	cout << agev[0] << " " << genderv[0] << endl;
	waitKey(0);*/
	return 0;
}
