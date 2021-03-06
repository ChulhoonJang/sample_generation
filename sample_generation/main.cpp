#include <stdlib.h>
#include <direct.h>
#include <iostream>
#include <Windows.h>
#include <opencv/highgui.h>
#include <opencv2\imgproc\imgproc.hpp>
#include "Shlwapi.h"

#include "annotation_class.h"

using namespace std;
using namespace cv;

vector<Point> polygon;
vector<vector<Point>> polygon_groups;

void import_annotation(FileStorage & f, string name){
	FileNode fn = f[name];
	FileNodeIterator it = fn.begin(), it_end = fn.end();

	polygon_groups.clear();
	for (int idx = 0; it != it_end; ++it, idx++){
		vector<Point> p;
		vector<int> x_val, y_val;
		(*it)["x"] >> x_val;
		(*it)["y"] >> y_val;
		for (int i = 0; i<(int)x_val.size(); i++){
			p.push_back(Point(x_val[i], y_val[i]));
		}
		if(!p.empty()) polygon_groups.push_back(p);
	}
}

void import_annotation(FileNode & fn, string name){
	FileNodeIterator it = fn.begin(), it_end = fn.end();

	polygon_groups.clear();
	for (int idx = 0; it != it_end; ++it, idx++){
		vector<Point> p;
		vector<int> x_val, y_val;
		(*it)["x"] >> x_val;
		(*it)["y"] >> y_val;
		for (int i = 0; i<(int)x_val.size(); i++){
			p.push_back(Point(x_val[i], y_val[i]));
		}
		if (!p.empty()) polygon_groups.push_back(p);
	}
}

void generate_polygon_mask(Mat src, Mat & dst, vector<Point> poly){
	Mat img_bin = Mat::zeros(src.size(), CV_8UC1);

	vector<vector<Point>> pPoly;
	pPoly.push_back(poly);
	fillPoly(img_bin, pPoly, Scalar(255, 255, 255));
	dst = img_bin.clone();
}

void generate_polygon_mask(Size sz, Mat & dst, vector<vector<Point>> polygon_groups){
	Mat img_bin = Mat::zeros(sz, CV_8UC1);
	fillPoly(img_bin, polygon_groups, Scalar(255, 255, 255));
	dst = img_bin.clone();
}

bool draw_annotations(const char * file_name, Size sz, Mat & dst, string attribute){
	FileStorage fr;
	if (!fr.open(file_name, FileStorage::READ)) return false;
		
	Mat img_mask;
	import_annotation(fr, attribute);
	generate_polygon_mask(sz, img_mask, polygon_groups);
	
	dst = img_mask.clone();
	return true;
}

bool draw_annotations(FileStorage & f, Size sz, Mat & dst, string attribute){
	Mat img_mask;
	import_annotation(f, attribute);
	generate_polygon_mask(sz, img_mask, polygon_groups);

	dst = img_mask.clone();
	return true;
}

void write_samples_on_csv(ofstream & csv_file, vector<Rect> & values, int n_class){
	for (vector<Rect>::iterator it = values.begin(); it != values.end(); it++){
		csv_file << n_class <<","<< it->x << "," << it->y << "," << it->width << "," << it->height << endl;
	}
}

int main(int argc, char *argv[], char *envp[])
{
	string root;
	string image_dir, annotation_dir;
	string saved_dir, gt_dir;
	//int patch_x(24), patch_y(24), roi_x(5), roi_y(5);
	//Rect rect_ego(74, 54, 156, 52);
	//Rect rect_ego(41, 47, 224, 63); // HG ego vehicle
	//264, 109

	if (argc < 2) return 0;
	root = argv[1];	
	//patch_x = atoi(argv[2]);
	//patch_y = atoi(argv[2]);

	image_dir = root + "/images";
	annotation_dir = root + "/annotations";
	saved_dir = root + "/samples";
	gt_dir = root + "/gt_images";

	_mkdir(saved_dir.c_str());
	_mkdir(gt_dir.c_str());	

	vector<int> params;
	params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	params.push_back(0);

	map<string, annotated_class*> ptr_class;
	annotated_class* c;

	c = new annotated_class(root, "marker", 1, CV_RGB(255,255,255));
	ptr_class.insert(pair<string, annotated_class*>("marker", c));

	c = new annotated_class(root, "vehicle", 2, CV_RGB(255, 0, 0));
	ptr_class.insert(pair<string, annotated_class*>("vehicle", c));	

	c = new annotated_class(root, "curb", 3, CV_RGB(0, 255, 0));
	ptr_class.insert(pair<string, annotated_class*>("curb", c));

	c = new annotated_class(root, "ego_vehicle", 3, CV_RGB(0, 0, 0));
	ptr_class.insert(pair<string, annotated_class*>("ego_vehicle", c));
	
	annotated_class free_space(root, "free_space", 0, CV_RGB(0, 0, 255), true);

	FileStorage f_info;
	string frame_yml = root + "/frame.yml";
	if (!f_info.open(frame_yml, FileStorage::READ)) return 0;
	FileNode f_info_n = f_info["frame"];
	FileNodeIterator f_it = f_info_n.begin();
	
	unsigned int n_c_0(1), n_c_1(1), r(0);
	cout << root << endl;
	for (; f_it != f_info_n.end();f_it++){
		int frame = (*f_it);

		// load image
		Mat img, img_debug;
		char image_path[200];
		sprintf_s(image_path, 200, "%s/%08d.jpg", image_dir.c_str(), frame);
		img = imread(image_path);
		if (img.data == NULL) continue;
		img_debug = img.clone();
		//rectangle(img_debug, rect_ego, CV_RGB(255, 0, 0));

		// open csv
		char temp[200];
		ofstream csv_file;
		sprintf_s(temp, 200, "%s/%08d.csv", saved_dir.c_str(), frame);
		//csv_file.open(temp, ios::out | ios::ate | ios::app);

		Mat img_mask = Mat::zeros(img.size(), CV_8UC1);
		Mat img_gt = Mat::zeros(img.size(), CV_8UC3);

		FileStorage f;
		char file_name[200];
		sprintf_s(file_name, 200, "%s/annotations/%08d.yml", root.c_str(), frame);
		if (!f.open(file_name, FileStorage::READ)) continue;		

		FileNode fn=f["attribute"];		
		for (FileNodeIterator it_fn = fn.begin(); it_fn != fn.end(); it_fn++){
			string attribute = (*it_fn);
			map<string, annotated_class*>::iterator it = ptr_class.find(attribute);
			draw_annotations(f, img.size(), it->second->img_mask, attribute);
			it->second->generate_gt_images();  

			//if (it->first == "marker"){
			//	Rect roi((patch_x - roi_x + 1) / 2, (patch_y - roi_y + 1) / 2, roi_x, roi_y);
			//	it->second->generate_samples(img, Size(patch_x, patch_y), 0.75, roi);
			//}
			//else{
			//	it->second->generate_samples(img, Size(patch_x, patch_y), 0.75);
			//}
			//write_samples_on_csv(csv_file, it->second->samples, it->second->n_class);
			if ((it->second)->img_mask.data != NULL) bitwise_or(img_mask, (it->second)->img_mask, img_mask);
			if ((it->second)->img_gt.data != NULL) bitwise_or(img_gt, (it->second)->img_gt, img_gt);
		}

		//Mat img_ego_mask = img_mask(rect_ego);		

		free_space.img_mask = ~img_mask.clone();
		free_space.generate_gt_images();
		bitwise_or(img_gt, free_space.img_gt, img_gt);

		//img_ego_mask = img_gt(rect_ego);
		//img_ego_mask = 255;

		//free_space.generate_random_samples(img, Size(patch_x, patch_y), 1.0, 1000);
		//write_samples_on_csv(csv_file, free_space.samples, free_space.n_class);

		char gt_img_file[200];
		sprintf_s(gt_img_file, "%s/%08d.png", gt_dir.c_str(), frame);
		imwrite(gt_img_file, img_gt, params);

		cout << image_path << endl;
	}
}