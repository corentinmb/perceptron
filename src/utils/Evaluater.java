package utils;

import java.util.ArrayList;
/*
 * 真实   正例数-A1  负例数-A2
 * 预测   正例数-B2  负例数-B2
 * TP = B2和A1的交集, 即预测正确的数目
 * FP = abs(B2-TP)
 * FN = abs(A1-TP)
 * TN = abs(A2-FP)
 * 四个数加起来=总样例数
 */

public class Evaluater {
	public static double getPrecisionV2(int correct, int predict) {
		return correct*1.0/predict;
	}
	public static double getPrecision(int TP, int FP) {
		return TP*1.0/(TP+FP);
	}
	public static double getRecallV2(int correct, int trued) {
		return correct*1.0/trued;
	}
	public static double getRecall(int TP, int FN) {
		return TP*1.0/(TP+FN);
	}
	
	public static double getFMeasure(double precision, double recall, int beita) {
		return (beita*beita+1)*precision*recall/(beita*beita*precision+recall);
	}
	
	public static double getPrecisionMicroAverage(ArrayList<Integer> TP, ArrayList<Integer> FP) {
		int sumTP = 0;
		for(Integer i:TP) 
			sumTP += i;
		int sumFP = 0;
		for(Integer i:FP)
			sumFP += i;
		return sumTP*1.0/(sumTP+sumFP);
	}
	
	public static double getRecallMicroAverage(ArrayList<Integer> TP, ArrayList<Integer> FN) {
		int sumTP = 0;
		for(Integer i:TP) 
			sumTP += i;
		int sumFN = 0;
		for(Integer i:FN)
			sumFN += i;
		return sumTP*1.0/(sumTP+sumFN);
	}
	
	public static double getPrecisionMacroAverage(ArrayList<Double> precisions) {
		double sum = 0;
		for(Double precision:precisions) {
			sum+=precision;
		}
		return sum/precisions.size();
	}
	
	public static double getRecallMacroAverage(ArrayList<Double> recalls) {
		double sum = 0;
		for(Double recall:recalls) {
			sum+=recall;
		}
		return sum/recalls.size();
	}
}
