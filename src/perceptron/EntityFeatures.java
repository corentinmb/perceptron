package perceptron;

import utils.Entity;
import utils.Tool;
import gnu.trove.TObjectDoubleHashMap;

/*
* Feature: whether the text of the current segment are consistent with its type.
*/
class EntityFeatures extends PerceptronFeatureFunction {

	public EntityFeatures(Perceptron perceptron) {
		super(perceptron);
	}

	@Override
	public void compute(PerceptronInputData x, PerceptronStatus status, PerceptronOutputData y, Object other,
			TObjectDoubleHashMap<String> map) {
		
		PerceptronInputData input = (PerceptronInputData)x;
		PerceptronOutputData output = (PerceptronOutputData)y;
		Tool tool = (Tool)other;

		int lastSegmentIndex = output.getLastSegmentIndex(status.tokenIndex);
		Entity lastSegment = output.segments.get(lastSegmentIndex);
	
		String text = lastSegment.text.toLowerCase();
		addFeature("E#WD_"+lastSegment.type+"_"+text, 1.0, status, y, map);
			
			
		
	}
	
	
	
}

