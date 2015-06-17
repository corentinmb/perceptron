package utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;

public class IoUtils {
	/*
	 * Delete all the files and directories in "dir".
	 */
	public static void clearDirectory(File dir) {  
	
	    File[] files = dir.listFiles();  
	    for (int i = 0; i < files.length; i++) {  
	        if (files[i].isFile()) {  
	            files[i].delete();  
	        } 
	        else {  
	        	clearDirectory(files[i]);  
	        	files[i].delete();
	        }  
	    }  
	    
	}  
	
	public static String readAFileAll(File file, String charset) {
		String contentString = null;
		try {
			InputStreamReader isr = new InputStreamReader(new FileInputStream(file), charset);
	    	StringBuilder sb = new StringBuilder();
	    	int ch = -1;
			while ((ch = isr.read()) != -1) {
				sb.append((char)ch);
			}    
			isr.close();
			contentString = new String(sb);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return contentString;
	}
}
