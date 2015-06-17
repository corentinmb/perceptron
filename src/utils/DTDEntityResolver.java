package utils;

import java.io.IOException;

import org.xml.sax.EntityResolver;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

public class DTDEntityResolver implements EntityResolver {
	private String dtdPath;
	public DTDEntityResolver(String dtdPath) {
		this.dtdPath = dtdPath;
	}
    @Override  
    public InputSource resolveEntity(String arg0, String arg1) throws SAXException, IOException {  
         return new InputSource(dtdPath); 
    }
   
} 
