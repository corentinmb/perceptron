package perceptron;

import gnu.trove.TDoubleArrayList;
import gnu.trove.TIntArrayList;
import gnu.trove.TObjectDoubleHashMap;
import gnu.trove.TObjectIntHashMap;

import java.io.Serializable;
import java.util.ArrayList;
import utils.Entity;
import cc.mallet.types.SparseVector;


public class Perceptron implements Serializable {

	private static final long serialVersionUID = 566743847777126825L;
	
	public SparseVector w; 
	
	public ArrayList<PerceptronFeatureFunction> featureFunctions; 
	
	public TObjectIntHashMap<String> featureAlphabet = new TObjectIntHashMap<>(); 
	
	public TIntArrayList d;
	
	public ArrayList<String> alphabet;
	public static final String EMPTY = "#N#";
	
	public Perceptron(ArrayList<String> alphabet, TIntArrayList d) {
		
		this.alphabet = new ArrayList<String>();
		for(int i=0;i<alphabet.size();i++) {
			this.alphabet.add(alphabet.get(i));
		}
		this.alphabet.add(0, Perceptron.EMPTY);		
		
		this.d = new TIntArrayList();
		this.d.add(1);
		for(int i=0;i<d.size();i++) {
			this.d.add(d.get(i));
		}
		
		w = new SparseVector();
	}
	
	public void buildFeatureAlphabet(ArrayList<PerceptronInputData> inputDatas, ArrayList<PerceptronOutputData> outputDatas, Object other) {
		try {
			
			for(int i=0;i<inputDatas.size();i++)  {
				for(int j=0;j<inputDatas.get(i).tokens.size();j++) { 
					PerceptronStatus status = new PerceptronStatus(null, j, 0); 
					f(inputDatas.get(i), status, outputDatas.get(i), other);
				}
			}

		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void trainPerceptron(int T, int beamSize, ArrayList<PerceptronInputData> input, ArrayList<PerceptronOutputData> output, Object other) {
		try {		
			for(int i=0;i<T;i++) {
				
				long startTime = System.currentTimeMillis();
				for(int j=0;j<input.size();j++) {
					PerceptronInputData x = input.get(j);
					PerceptronOutputData y = output.get(j);

					PerceptronStatus status = beamSearch(x, y, true, beamSize, other);
					
					if(!status.z.isIdenticalWith(x, y, status)) {
						SparseVector fxy  = f(x, status, y, other).sv;
						SparseVector fxz = f(x, status, status.z, other).sv;
						SparseVector temp = fxy.vectorAdd(fxz, -1);
						w = w.vectorAdd(temp, 1);

					}
				}


				long endTime = System.currentTimeMillis();
				System.out.println((i+1)+" train finished with "+(endTime-startTime)+" ms");

			}
			System.out.println("achieve max training times, quit");
		} catch(Exception e) {
			e.printStackTrace();
		}

		normalizeWeight();
		return;
	}
	
	public PerceptronStatus beamSearch(PerceptronInputData x,
			PerceptronOutputData y, boolean isTrain, int beamSize, Object other)
			throws Exception {
		ArrayList<ArrayList<PerceptronOutputData>> beams = new ArrayList<ArrayList<PerceptronOutputData>>();
		for(int i=0;i<x.tokens.size();i++)
			beams.add(new ArrayList<PerceptronOutputData>());
		
		
		for(int i=0;i<x.tokens.size();i++) {
			ArrayList<PerceptronOutputData> buf = new ArrayList<PerceptronOutputData>();
			for(int t=0;t<this.alphabet.size();t++) {
				if(i==0) {
					buf.add(PerceptronOutputData.append(null, alphabet.get(t), x, 0, 0));
					continue;
				}
				for(int dd=1;dd<=this.d.get(t);dd++) {
					if(i-dd>=0) {
						for(int yy=0;yy<beams.get(i-dd).size();yy++) {
							int k = i-dd+1;
							buf.add(PerceptronOutputData.append(beams.get(i-dd).get(yy), alphabet.get(t), x, k, i));
						}
					} else if(i-dd==-1){ 
						buf.add(PerceptronOutputData.append(null, alphabet.get(t), x, 0, i));
						break;
					}
				}
			}
			
			PerceptronStatus statusKBest = new PerceptronStatus(null, i, 1);
			kBest(x, statusKBest, beams.get(i), buf, beamSize, other);
			// early update
			if(isTrain) {
				int m=0;
				for(;m<beams.get(i).size();m++) {
					if(beams.get(i).get(m).isIdenticalWith(x, y, statusKBest)) {
						break;
					}
				}
				if(m==beams.get(i).size() && isAlignedWithGold(beams.get(i).get(0), y, i)) {
					PerceptronStatus returnType = new PerceptronStatus(beams.get(i).get(0), i, 1);
					return returnType;
				}
			}

			
		}
		
		PerceptronStatus returnType = new PerceptronStatus(beams.get(x.tokens.size()-1).get(0), x.tokens.size()-1, 3);
		return returnType;
	}
	
	public void kBest(PerceptronInputData x, PerceptronStatus status, ArrayList<PerceptronOutputData> beam, ArrayList<PerceptronOutputData> buf, int beamSize, Object other)throws Exception {
		// compute all the scores in the buf
		TDoubleArrayList scores = new TDoubleArrayList();
		for(PerceptronOutputData y:buf) {
			FReturnType ret = f(x,status,y, other);
			
			scores.add(w.dotProduct(ret.sv));

		}
		
		// assign k best to the beam, and note that buf may be more or less than beamSize.
		int K = buf.size()>beamSize ? beamSize:buf.size();
		PerceptronOutputData[] temp = new PerceptronOutputData[K];
		Double[] tempScore = new Double[K];
		for(int i=0;i<buf.size();i++) {
			for(int j=0;j<K;j++) {
				if(temp[j]==null || scores.get(i)>tempScore[j]) {
					if(temp[j] != null) {
						for(int m=K-2;m>=j;m--) {
							temp[m+1] = temp[m];
							tempScore[m+1] = tempScore[m];
						}
					}
					
					temp[j] = buf.get(i);
					tempScore[j] = scores.get(i);
					break;
				}
			}
		}
		
		beam.clear();
		for(int i=0;i<K;i++) {
			beam.add(temp[i]);
		}
		
		return;
	}
	
	public static boolean isAlignedWithGold(PerceptronOutputData predict, PerceptronOutputData gold, int tokenIndex) {
		PerceptronOutputData predict1 = (PerceptronOutputData)predict;
		PerceptronOutputData gold1 = (PerceptronOutputData)gold;
		
		Entity predictLastSeg = predict1.segments.get(predict1.getLastSegmentIndex(tokenIndex));
		Entity goldLastSeg = gold1.segments.get(gold1.getLastSegmentIndex(tokenIndex));
		if(predictLastSeg.end==goldLastSeg.end)
			return true;
		else return false;
	}
	
	public void normalizeWeight() {
		double norm1 = w.twoNorm();
		for(int j=0;j<w.getIndices().length;j++) {
			w.setValueAtLocation(j, w.valueAtLocation(j)/norm1);
		}
	}
	
	public class FReturnType {
		public SparseVector sv;
		public FReturnType(SparseVector sv) {
			super();
			this.sv = sv;
		}
	}
	
	public FReturnType f(PerceptronInputData x, PerceptronStatus status, PerceptronOutputData y, Object other) throws Exception {	
		
		if(y.isGold) {
			if(status.step==0) { 
				TObjectDoubleHashMap<String> map = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions.get(j);
					featureFunction.compute(x, status, y, other, map);
				}
				y.featureVectors.add(hashMapToSparseVector(map));
				
				return new FReturnType(y.featureVectors.get(status.tokenIndex));
			} else {
				
				return new FReturnType(y.featureVectors.get(status.tokenIndex));
			}
			
		} else {
			
			TObjectDoubleHashMap<String> map = new TObjectDoubleHashMap<>();
			for(int j=0;j<featureFunctions.size();j++) {
				PerceptronFeatureFunction featureFunction = featureFunctions.get(j);
				featureFunction.compute(x, status, y, other, map);
			}
			y.featureVector = hashMapToSparseVector(map);
			return new FReturnType(y.featureVector);
			 
			
		}

	}
	
	public SparseVector hashMapToSparseVector(TObjectDoubleHashMap<String> map) {
		TIntArrayList featureIndices = new TIntArrayList();
		TDoubleArrayList featureValues = new TDoubleArrayList();
		String[] keys = map.keys( new String[ map.size() ] );
		for(String featureName:keys) {
			featureIndices.add(this.featureAlphabet.get(featureName));
    		featureValues.add(map.get(featureName));
		}
		
        int[] featureIndicesArr = new int[featureIndices.size()];
        double[] featureValuesArr = new double[featureValues.size()];
        for (int index = 0; index < featureIndices.size(); index++) {
        	featureIndicesArr[index] = featureIndices.get(index);
        	featureValuesArr[index] = featureValues.get(index);
        }
		
        SparseVector fxy = new SparseVector(featureIndicesArr, featureValuesArr, false);
        return fxy;
	}
	
	public void setFeatureFunction(ArrayList<PerceptronFeatureFunction> featureFunctions) {
		this.featureFunctions = featureFunctions;
	}
}
