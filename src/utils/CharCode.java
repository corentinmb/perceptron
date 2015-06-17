package utils;

public class CharCode {
	/*
	 * check whether the characters in s are all Chinese
	 * CJK Unified Ideographs
	 */
	public static boolean isAllChinese(String s) {
		int min = 0x4e00;
		int max = 0x9fa5;
		for(int i=0;i<s.length();i++) {
			int c = (int)s.charAt(i);
			if(c<min || c>max)
				return false;
		}
		
		return true;
	}
	
	
	public static boolean isSeperator(String word) {
		String[] seperator = {"\uFF0C", "\u3002", "\uFF01", "\uFF1F", "\u3001", "\uFF1A", "\uFF1B"};
		for(int i=0;i<seperator.length;i++) {
			if(word.equals(seperator[i]))
					return true;
		}
		return false;
	}
	
	// english or chinese
	public static boolean isNumber(String s) {
		int c = (int)s.charAt(0);
		if(c>=0xFF10 && c<=0xFF19)
			return true;
		else if(c>=0x0030 && c<=0x0039)
			return true;
		
		return false;
	}
	// english or chinese
	public static boolean isNumber(char c) {
		if(c>=0xFF10 && c<=0xFF19)
			return true;
		else if(c>=0x0030 && c<=0x0039)
			return true;
		
		return false;
	}
	// english
	public static boolean isUpperCase(char c) {
		if(c>=0x0041 && c<=0x005A)
			return true;
		else return false;
	}
	public static boolean isLowerCase(char c) {
		if(c>=0x0061 && c<=0x007A)
			return true;
		else return false;
	}
	public static boolean isEnAlpha(char c) {
		if((c>=0x0041 && c<=0x005A) || (c>=0x0061 && c<=0x007A))
			return true;
		else return false;
	}
	// chinese
	public static boolean isChCharacter(String s) {
		int minUpper = 0xFF21; int minLower = 0xFF41;
		int maxUpper = 0xFF3A; int maxLower = 0xFF5A;
		int c = (int)s.charAt(0);
		if((c>=minUpper && c<=maxUpper) || (c>=minLower && c<=maxLower))
			return true;
		
		return false;
	}
	
	public static boolean isWhiteSpace(String s) {
		int w1 = 0x0020; int w2 = 0x3000;
		int c = (int)s.charAt(0);
		if(c == w1 || c==w2 )
			return true;
		return false;
	}
	/*public static boolean isPunctuation(String s) {
		String[] temp = {"\u3001", "\u3002", "\u300A", "\u300B", "", "\uFF1A", "\uFF1B"};
		int min = 0x3000;
		int max = 0x303f;
		int c = (int)s.charAt(0);
		if(c<min || c>max)
			return false;
		
		return true;
	}
	
	
	 * Halfwidth and Fullwidth Forms
	 
	public static boolean isHalfFullForm(String s) {
		int min = 0xFF01;
		int max = 0xFFEE;
		int c = (int)s.charAt(0);
		if(c<min || c>max)
			return false;
		
		return true;
	}*/
}
