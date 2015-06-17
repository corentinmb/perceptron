package joint;

import gnu.trove.TDoubleArrayList;
import gnu.trove.TIntArrayList;
import gnu.trove.TObjectDoubleHashMap;
import gnu.trove.TObjectIntHashMap;

import java.io.Serializable;
import java.util.ArrayList;

import utils.Entity;
import utils.RelationEntity;
import cc.mallet.types.SparseVector;



public class Perceptron implements Serializable{
	private static final long serialVersionUID = -6988787945862648789L;
	
	public static final String EMPTY = "#N#"; 
	
	public SparseVector w1; 
	public SparseVector w2; 
	
	public ArrayList<PerceptronFeatureFunction> featureFunctions1; 
	public ArrayList<PerceptronFeatureFunction> featureFunctions2; 

	public ArrayList<String> alphabet1;
	public ArrayList<String> alphabet2;
	
	public TObjectIntHashMap<String> featureAlphabet; 
	
	private TIntArrayList d;
	
		
	public Perceptron(ArrayList<String> alphabet1, ArrayList<String> alphabet2, TIntArrayList d) {
		
		this.alphabet1 = new ArrayList<String>();
		for(int i=0;i<alphabet1.size();i++) {
			this.alphabet1.add(alphabet1.get(i));
		}
		this.alphabet1.add(0, Perceptron.EMPTY);
	
	
		this.alphabet2 = new ArrayList<String>();
		for(int i=0;i<alphabet2.size();i++) {
			this.alphabet2.add(alphabet2.get(i));
		}
		
		
		this.featureAlphabet = new TObjectIntHashMap<>();
		
		this.d = new TIntArrayList();
		this.d.add(1);
		for(int i=0;i<d.size();i++) {
			this.d.add(d.get(i));
		}
		
		w1 = new SparseVector();
		w2 = new SparseVector();
	}
	
	public void setFeatureFunction(ArrayList<PerceptronFeatureFunction> featureFunctions1, ArrayList<PerceptronFeatureFunction> featureFunctions2) {
		this.featureFunctions1 = featureFunctions1;
		this.featureFunctions2 = featureFunctions2;
	}
	
	public void buildFeatureAlphabet(ArrayList<PerceptronInputData> inputDatas, ArrayList<PerceptronOutputData> outputDatas, Object other) {
		try {

			for(int i=0;i<inputDatas.size();i++)  {
				for(int j=0;j<inputDatas.get(i).tokens.size();j++) { 
					PerceptronStatus status = new PerceptronStatus(null, j, 0); 
					FReturnType ret = f(inputDatas.get(i), status, outputDatas.get(i), other);
				}
			}

		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void normalizeWeight() {
		// norm
		double norm1 = w1.twoNorm();
		for(int j=0;j<w1.getIndices().length;j++) {
			w1.setValueAtLocation(j, w1.valueAtLocation(j)/norm1);
		}
		double norm2 = w2.twoNorm();
		for(int j=0;j<w2.getIndices().length;j++) {
			w2.setValueAtLocation(j, w2.valueAtLocation(j)/norm2);
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
						
						if(status.step==1) {
							SparseVector fxy  = f(x, status, y, other).sv1;
							SparseVector fxz = f(x, status, status.z, other).sv1;
							SparseVector temp = fxy.vectorAdd(fxz, -1);
							w1 = w1.vectorAdd(temp, 1);
						}
						else if(status.step==2 || status.step==3) {
							FReturnType rtFxy = f(x, status, y, other);
							FReturnType rtFxz = f(x, status, status.z, other);
							
							SparseVector fxy1  = rtFxy.sv1;
							SparseVector fxz1 = rtFxz.sv1;
							SparseVector temp1 = fxy1.vectorAdd(fxz1, -1);
							w1 = w1.vectorAdd(temp1, 1);
							
							SparseVector fxy2  = rtFxy.sv2;
							SparseVector fxz2 = rtFxz.sv2;
							SparseVector temp = fxy2.vectorAdd(fxz2, -1);
							w2 = w2.vectorAdd(temp, 1);
						}
						else
							throw new Exception();
						
						
					}
				}
				
				long endTime = System.currentTimeMillis();
				System.out.println((i+1)+" train finished with "+(endTime-startTime)+" ms");
				

			}
			System.out.println("achieve max training times, quit");
		} catch(Exception e) {
			e.printStackTrace();
		}
		// norm
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
			for(int t=0;t<this.alphabet1.size();t++) {
				if(i==0) {
					buf.add(PerceptronOutputData.append(null, alphabet1.get(t), x, 0, 0));
					continue;
				}
				for(int dd=1;dd<=this.d.get(t);dd++) {
					if(i-dd>=0) {
						for(int yy=0;yy<beams.get(i-dd).size();yy++) {
							int k = i-dd+1;
							buf.add(PerceptronOutputData.append(beams.get(i-dd).get(yy), alphabet1.get(t), x, k, i));
						}
					} else if(i-dd==-1){ 
						buf.add(PerceptronOutputData.append(null, alphabet1.get(t), x, 0, i));
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

			PerceptronStatus statusKBest1 = new PerceptronStatus(null, i, 2);
			for(int j=i-1;j>=0;j--) {
				buf.clear();
				for(int yy=0;yy<beams.get(i).size();yy++) {
					PerceptronOutputData yInBeam = (PerceptronOutputData)beams.get(i).get(yy);
					buf.add(yInBeam);
					
					Entity entityI = null;
					Entity entityJ = null;
					for(int m=0;m<yInBeam.segments.size();m++) {
						Entity entity = yInBeam.segments.get(m);
						if(entity.type.equals(Perceptron.EMPTY))
							continue;
						if(entity.end == i)
							entityI = entity;
						if(entity.end == j)
							entityJ = entity;
						if(entityI!=null && entityJ!=null)
							break;
					}
					if(entityI!=null && entityJ!=null ) {
					
						for(int r=0;r<alphabet2.size();r++) {
							
							PerceptronOutputData ret = new PerceptronOutputData(false);
							for(int m=0;m<yInBeam.segments.size();m++) {
								ret.segments.add(yInBeam.segments.get(m));
							}
							for(RelationEntity relation:yInBeam.relations) {
								ret.relations.add(relation);
							}
							RelationEntity relation = new RelationEntity(alphabet2.get(r), entityI, entityJ);
							ret.relations.add(relation);
							buf.add(ret);
							
						}
					}
				}

				kBest(x, statusKBest1, beams.get(i), buf, beamSize, other);
			}
			// early update
			if(isTrain) {
				int m=0;
				for(;m<beams.get(i).size();m++) {
					if(beams.get(i).get(m).isIdenticalWith(x, y, statusKBest1)) {
						break;
					}
				}
				if(m==beams.get(i).size() && isAlignedWithGold(beams.get(i).get(0), y, i)) {
					PerceptronStatus returnType = new PerceptronStatus(beams.get(i).get(0), i, 2);
					return returnType;
				}
			}
		}
		
		PerceptronStatus returnType = new PerceptronStatus(beams.get(x.tokens.size()-1).get(0), x.tokens.size()-1, 3);
		return returnType;
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

	public void kBest(PerceptronInputData x, PerceptronStatus status, ArrayList<PerceptronOutputData> beam, ArrayList<PerceptronOutputData> buf, int beamSize, Object other)throws Exception {
		// compute all the scores in the buf
		TDoubleArrayList scores = new TDoubleArrayList();
		for(PerceptronOutputData y:buf) {
			FReturnType ret = f(x,status,y, other);
			if(status.step==1) {
				scores.add(w1.dotProduct(ret.sv1));
			} else if(status.step==2) {
				scores.add(w1.dotProduct(ret.sv1)+w2.dotProduct(ret.sv2));
			} else 
				throw new Exception();
			
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
	
	public class FReturnType {
		public SparseVector sv1;
		public SparseVector sv2;
		public FReturnType(SparseVector sv1, SparseVector sv2) {
			super();
			this.sv1 = sv1;
			this.sv2 = sv2;
		}
	}
	
	protected FReturnType f(PerceptronInputData x, PerceptronStatus status, PerceptronOutputData y, Object other) throws Exception {	
		
		if(y.isGold) {
			if(status.step==0) { // initialize the feature vectors of gold output
				TObjectDoubleHashMap<String> map1 = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions1.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions1.get(j);
					featureFunction.compute(x, status, y, other, map1);
				}
				y.featureVectors1.add(hashMapToSparseVector(map1));
				
				TObjectDoubleHashMap<String> map2 = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions2.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions2.get(j);
					featureFunction.compute(x, status, y, other, map2);
				}
				y.featureVectors2.add(hashMapToSparseVector(map2));
				return new FReturnType(y.featureVectors1.get(status.tokenIndex), y.featureVectors2.get(status.tokenIndex));
			} else if(status.step==1) {
				return new FReturnType(y.featureVectors1.get(status.tokenIndex), null);
			} else if(status.step==2 || status.step==3) {
				return new FReturnType(y.featureVectors1.get(status.tokenIndex), y.featureVectors2.get(status.tokenIndex));
			} else
				throw new Exception();
			
		} else {
			if(status.step==1) {
				TObjectDoubleHashMap<String> map = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions1.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions1.get(j);
					featureFunction.compute(x, status, y, other, map);
				}
				y.featureVector1 = hashMapToSparseVector(map);
				return new FReturnType(y.featureVector1, null);
			} else if(status.step==2 || status.step==3) {
				TObjectDoubleHashMap<String> map1 = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions1.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions1.get(j);
					featureFunction.compute(x, status, y, other, map1);
				}
				y.featureVector1 = hashMapToSparseVector(map1);
				
				TObjectDoubleHashMap<String> map = new TObjectDoubleHashMap<>();
				for(int j=0;j<featureFunctions2.size();j++) {
					PerceptronFeatureFunction featureFunction = featureFunctions2.get(j);
					featureFunction.compute(x, status, y, other, map);
				}
				y.featureVector2 = hashMapToSparseVector(map);
				return new FReturnType(y.featureVector1, y.featureVector2);
			} else
				throw new Exception();
			
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
	
	
}
