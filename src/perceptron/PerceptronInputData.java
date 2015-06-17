package perceptron;

import gnu.trove.TIntArrayList;

import java.io.Serializable;
import java.util.ArrayList;

import utils.Sentence;


public class PerceptronInputData implements Serializable{
	
	private static final long serialVersionUID = 4107472666142847362L;
	
	public ArrayList<String> tokens = new ArrayList<String>();
	
	public TIntArrayList offset;
	
	public Sentence sentInfo; 
	
	public PerceptronInputData() {
		offset = new TIntArrayList();
	}
	
	@Override
	public String toString() {
		String s = tokens.get(0);
		for(int i=1;i<tokens.size();i++)
			s += ", "+tokens.get(i);
		return s;
	}

}
