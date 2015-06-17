package joint;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;

import utils.Entity;
import utils.RelationEntity;
import cc.mallet.types.SparseVector;



public class PerceptronOutputData implements Serializable{
	
	private static final long serialVersionUID = 8669519647030994555L;

	public ArrayList<SparseVector> featureVectors1;
	public ArrayList<SparseVector> featureVectors2;

	public SparseVector featureVector1;
	public SparseVector featureVector2;

	public boolean isGold; 
	
	public ArrayList<Entity> segments; 
	public HashSet<RelationEntity> relations;
	
	public PerceptronOutputData(boolean isGold) {
		this.isGold = isGold;
		
		featureVectors1 = new ArrayList<>();
		featureVectors2 = new ArrayList<>();
		
		segments = new ArrayList<Entity>();
		relations = new HashSet<RelationEntity>();
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
		if(status.step==1) {
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

		if(status.step==2 || status.step==3) {
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
			
			HashSet<RelationEntity> otherRelation = new HashSet<RelationEntity>(); // other is a gold output data
			for(RelationEntity relation:other1.relations) {
				if(relation.entity1.end<=status.tokenIndex && relation.entity2.end<=status.tokenIndex)
					otherRelation.add(relation);
			}
			if(!relations.equals(otherRelation))
				return false;
			
		}
		
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
		// copy relation
		for(RelationEntity relation:y.relations) {
			ret.relations.add(relation);
		}
		return ret;
	}
	
	@Override
	public String toString() {
		String s = "segments: "+segments.get(0);
		for(int i=1;i<segments.size();i++)
			s += ", "+segments.get(i);
		s+="\n";
		s+="relations: ";
		int i=0;
		for(RelationEntity relation:relations) {
			if(i!=0)
				s+=", "+relation;
			else
				s+=relation;
			i++;
		}
		s+="\n";
		return s;
	}
}
