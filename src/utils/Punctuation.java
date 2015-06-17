package utils;

import java.util.HashSet;

public class Punctuation {
	private static HashSet<Character> english;
	static {
		english =  new HashSet<Character>();
		english.add('`');english.add('~');english.add('!');english.add('@');english.add('#');english.add('$');
		english.add('%');english.add('&');english.add('*');english.add('(');english.add(')');english.add('-');
		english.add('_');english.add('+');english.add('=');english.add('{');english.add('}');english.add('|');
		english.add('[');english.add(']');english.add('\\');english.add(':');english.add(';');english.add('\'');
		english.add('"');english.add('<');english.add('>');english.add(',');english.add('.');english.add('?');
		english.add('/');
	}
	
		
	public static boolean isEnglishPunc(Character ch) {
		return english.contains(ch);
	}
}
