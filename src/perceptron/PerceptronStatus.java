package perceptron;


public class PerceptronStatus {
	
	public PerceptronOutputData z;
	
	public int tokenIndex;
	
	public int step;
	
	public PerceptronStatus(PerceptronOutputData z, int tokenIndex, int step) {
		super();
		this.z = z;
		this.tokenIndex = tokenIndex;
		this.step = step;
	}
}
