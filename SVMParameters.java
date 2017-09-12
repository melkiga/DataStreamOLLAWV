package vcu.edu.datastreamlearning.ollawv;

public class SVMParameters {
	/**
	 * SVM Hyper-parameter C, double
	 */
	private double C;
	/**
	 * Gaussian RBF Kernel Gamma parameter, double
	 */
	private double gamma;
	/**
	 * Integer indicating whether to use the bias term or not, int
	 */
	private int betta;
	/**
	 * Percentage of data that the model should use for training, double
	 */
	private double epochs;
	/**
	 * Size of margin based on C value, double between (0,1]
	 */
	private double tol;
	
	public SVMParameters(){
		
	}

	public double getC() {
		return C;
	}

	public void setC(double c) {
		C = c;
	}

	public double getGamma() {
		return gamma;
	}

	public void setGamma(double gamma) {
		this.gamma = gamma;
	}

	public int getBetta() {
		return betta;
	}

	public void setBetta(int betta) {
		this.betta = betta;
	}

	public double getEpochs() {
		return epochs;
	}

	public void setEpochs(double epochs) {
		this.epochs = epochs;
	}

	public double getTol() {
		return tol;
	}

	public void setTol(double tol) {
		this.tol = tol;
	}
}
