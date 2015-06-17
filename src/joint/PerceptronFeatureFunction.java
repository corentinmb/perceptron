package joint;

import gnu.trove.TObjectDoubleHashMap;

import java.io.Serializable;


public abstract class PerceptronFeatureFunction implements Serializable{

	private static final long serialVersionUID = 1869086926811075989L;
	public Perceptron perceptron;
	
	public PerceptronFeatureFunction(Perceptron perceptron) {
		this.perceptron = perceptron;
	}
	
	public void compute(PerceptronInputData x, PerceptronStatus status, PerceptronOutputData y, Object other, 
			TObjectDoubleHashMap<String> map) {
		return ;
	}
	
	public void addFeature(String name, double value, PerceptronStatus status, PerceptronOutputData y,
			TObjectDoubleHashMap<String> map) {


		if(status.step == 0 ) { 
			if(!perceptron.featureAlphabet.containsKey(name)) {
				perceptron.featureAlphabet.put(name, perceptron.featureAlphabet.size());
			}
			map.put(name, value);
						
			return ;
		}
		
		if(perceptron.featureAlphabet.containsKey(name)) {
			map.put(name, value);
		}
		

		return ;
	}
}
