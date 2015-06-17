package perceptron;

import java.io.Serializable;
import java.util.ArrayList;

import utils.Entity;
import cc.mallet.types.SparseVector;


public class PerceptronOutputData implements Serializable{
	
	private static final long serialVersionUID = 8669519647030994555L;

	public ArrayList<SparseVector> featureVectors = new ArrayList<>();

	public SparseVector featureVector;

	public boolean isGold; 
	
	public ArrayList<Entity> segments; 
	
	public PerceptronOutputData(boolean isGold) {
		
		this.isGold = isGold;
		segments = new ArrayList<Entity>();
	}
	
	
	public int getLastSegmentIndex(int tokenIndex) {
		if(isGold) {
			int i=0;
			Entity thisSegment = null;
			do {
				thisSegment = segments.get(i);
				i++;
			}while(tokenIndex>thisSegment.end);
			return i-1;
		} else {
			return segments.size()-1;
		}
	}
	
	public boolean isIdenticalWith(PerceptronInputData input, PerceptronOutputData other, PerceptronStatus status) {
		PerceptronOutputData other1 = (PerceptronOutputData)other;
		
		int i=0;
		Entity thisSegment = null;
		Entity OtherSegment = null;
		do {
			thisSegment = segments.get(i);
			OtherSegment = other1.segments.get(i);
			if(!thisSegment.equals(OtherSegment))
				return false;
			
			i++;
		}while(status.tokenIndex>thisSegment.end);
		
		return true;
		
	}
	
	
	public static PerceptronOutputData append(PerceptronOutputData yy, String t, PerceptronInputData xx, int k, int i) {
		PerceptronInputData x = (PerceptronInputData)xx;
		PerceptronOutputData y = (PerceptronOutputData)yy;
		PerceptronOutputData ret = new PerceptronOutputData(false);
		if(yy == null) {
			// append segment
			int segmentOffset = x.offset.get(k);
			String segmentText = "";
			for(int m=k;m<=i;m++) {
				int whitespaceToAdd = x.offset.get(m)-(segmentOffset+segmentText.length());
				if(whitespaceToAdd>0) {
					for(int j=0;j<whitespaceToAdd;j++)
						segmentText += " ";
				}
				segmentText += x.tokens.get(m);
			}
			Entity segment = new Entity(null, t, segmentOffset, segmentText, null);
			segment.start = k;
			segment.end = i;
			ret.segments.add(segment);
			return ret;
		}
			
		// copy segment
		for(int m=0;m<y.segments.size();m++) {
			ret.segments.add(y.segments.get(m));
		}
		// append segment
		int segmentOffset = x.offset.get(k);
		String segmentText = "";
		for(int m=k;m<=i;m++) {
			int whitespaceToAdd = x.offset.get(m)-(segmentOffset+segmentText.length());
			if(whitespaceToAdd>0) {
				for(int j=0;j<whitespaceToAdd;j++)
					segmentText += " ";
			}	
			segmentText += x.tokens.get(m);
		}
		Entity segment = new Entity(null, t, segmentOffset, segmentText, null);
		segment.start = k;
		segment.end = i;
		ret.segments.add(segment);
		
		return ret;
	}
	
	
	
	@Override
	public String toString() {
		String s = "segments: "+segments.get(0);
		for(int i=1;i<segments.size();i++)
			s += ", "+segments.get(i);
		s+="\n";
		
		return s;
	}
}
