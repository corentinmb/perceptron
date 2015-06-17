package joint;

import gnu.trove.TIntArrayList;

import java.io.Serializable;
import java.util.ArrayList;

import utils.Sentence;


public class PerceptronInputData implements Serializable{
	
	private static final long serialVersionUID = 4107472666142847362L;
	
	public ArrayList<String> tokens;
	
	public TIntArrayList offset;  
	
	public Sentence sentInfo; 
	
	public PerceptronInputData() {
		tokens = new ArrayList<String>();
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
