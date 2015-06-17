package utils;

public class Segment {
	public String word;
	public int begin; // the first position of this segment
	public int end; // the last+1 position of this segment, consistent with stanford.
	public Segment(String word, int begin, int end) {
		super();
		this.word = word;
		this.begin = begin;
		this.end = end;
	}
	
	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return word+" "+begin+" "+end;
	}
	
}
