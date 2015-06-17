package utils;

import java.io.Serializable;
import java.util.List;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.Tree;

public class Sentence implements Serializable{

	private static final long serialVersionUID = 7947631507213407579L;
	public String text;
	public int offset; 
	public int length;
	public List<CoreLabel> tokens;
	public Tree root;
	public List<Tree> leaves;
	public SemanticGraph depGraph;
	
}
