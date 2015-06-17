package joint;


import utils.Entity;
import utils.RelationEntity;
import utils.Tool;

import gnu.trove.TObjectDoubleHashMap;


public class RelationFeatures extends PerceptronFeatureFunction {
	public RelationFeatures(Perceptron perceptron) {
		super(perceptron);
	}
	
	@Override
	public void compute(PerceptronInputData x, PerceptronStatus status,
			PerceptronOutputData y, Object other,
			TObjectDoubleHashMap<String> map) {
		PerceptronInputData input = (PerceptronInputData)x;
		PerceptronOutputData output = (PerceptronOutputData)y;
		Tool tool = (Tool)other;
		
		
		int lastSegmentIndex = output.getLastSegmentIndex(status.tokenIndex);
		Entity latter = output.segments.get(lastSegmentIndex);
		
		if(latter.type.equals(Perceptron.EMPTY))
			return;
		
		for(int index=lastSegmentIndex-1;index>=0;index--) {
			Entity former = output.segments.get(index);
			if(former.type.equals(Perceptron.EMPTY))
				continue;
			
			String type = Perceptron.EMPTY;
			if(output.relations.contains(new RelationEntity("CID", former, latter)))
				type = "CID";
			
			addFeature("#EN_"+type+"_"+former.text.toLowerCase()+"_"+latter.text.toLowerCase(), 1.0, status, y, map);
			addFeature("#EN_"+type+"_"+latter.text.toLowerCase()+"_"+former.text.toLowerCase(), 1.0, status, y, map);
			
		}
			
			
		
	}
	

}
