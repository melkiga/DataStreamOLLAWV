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
	
	public SVMParameters(){
		this.C = 1.0;
		this.gamma = 0.5;
		this.betta = 1;
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
}
