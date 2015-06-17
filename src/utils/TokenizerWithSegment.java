package utils;

import java.util.ArrayList;



public class TokenizerWithSegment {
	/*
	 * Given some ordered Segments("word" doesn't need to be filled), tokenize "s" with whitespace 
	 * and keep the characters in the given segments together.
	 * Keep the "offset", "begin" and "end" in the same computational method.
	 */
	public static ArrayList<Segment> tokenize(int offset, String s, ArrayList<Segment> given) {
		// We assume that the segments are not overlapped, so if any, we get rid of the one with small range.
		ArrayList<Segment> toBeRemoved = new ArrayList<>();
		for(int i=0;i<given.size();i++) {
			for(int j=0;j<i;j++) {
				if(given.get(i).begin<=given.get(j).begin && given.get(i).end>=given.get(j).end) // i is bigger
					toBeRemoved.add(given.get(j));
				else if(given.get(i).begin>=given.get(j).begin && given.get(i).end<=given.get(j).end) // j is bigger
					toBeRemoved.add(given.get(i));
			}
		}
		for(Segment remove:toBeRemoved) {
			given.remove(remove);
		}
		// begin tokenization
		ArrayList<Segment> ret = new ArrayList<Segment>();
		char[] chs = s.toCharArray();
		StringBuilder sb = new StringBuilder();
		int from = 0;
		Segment temp = null;
		for(int i=0;i<chs.length; ) {
			Character ch = chs[i];
			if((temp=getSegment(from, offset, given))==null) { // not in any given segment
				if(ch==' ') {
					if(sb.length()!=0) {
						ret.add(new Segment(new String(sb), offset-sb.length(), offset));
						sb.delete(0, sb.length());
					}
				} else {
					if(Punctuation.isEnglishPunc(ch)) {
						if(sb.length()!=0) {
							ret.add(new Segment(new String(sb), offset-sb.length(), offset));
							sb.delete(0, sb.length());
						}
						ret.add(new Segment(ch.toString(), offset, offset+1)); 
					} else {
						sb.append(ch);
					}
				}
				offset++;
				i++;
			} else { 
				if(sb.length()!=0) {
					ret.add(new Segment(new String(sb), offset-sb.length(), offset));
					sb.delete(0, sb.length());
				}
				try {
				ret.add(new Segment(s.substring(i, i+temp.end-temp.begin), temp.begin, temp.end));
				} catch(Exception e) {
					e.printStackTrace();
				}
				from++;
				offset = temp.end;
				i+=temp.end-temp.begin;
			}
			
				
		}
		
		if(sb.length()!=0) {
			ret.add(new Segment(new String(sb), offset-sb.length(), offset));
			sb.delete(0, sb.length());
		}
		
		return ret;
	}
	
	// From the "from" segment, get the segment which contains the offset
	private static Segment getSegment(int from , int offset, ArrayList<Segment> given) {
		for(int i=from;i<given.size();i++) {
			if(offset>= given.get(i).begin && offset<given.get(i).end) {
				return given.get(i);
			}
		}
		return null;
	}
}
