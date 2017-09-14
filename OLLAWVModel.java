package vcu.edu.datastreamlearning.ollawv;

import java.util.List;

public class OLLAWVModel {
	/**
	 * Alpha values
	 */
	private List<Double> alphas;
	/**
	 * Support vector indices
	 */
	private List<Integer> Ind;
	/**
	 * Number of support vectors.
	 */
	private int svnumber;
	/**
	 * Bias term
	 */
	private double bias;
	/**
	 * SVM Hyper-parameters
	 */
	private SVMParameters params;
	
	/**
	 * Constructor for OLLAWV model
	 */
	public OLLAWVModel(SVMParameters params){
		this.setParams(params);
		setAlphas(null);
		setInd(null);
		setSvnumber(0);
		setBias(0.0);
	}

	public SVMParameters getParams() {
		return params;
	}

	public void setParams(SVMParameters params) {
		this.params = params;
	}

	public double getBias() {
		return bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public int getSvnumber() {
		return svnumber;
	}

	public void setSvnumber(int svnumber) {
		this.svnumber = svnumber;
	}

	public List<Integer> getInd() {
		return Ind;
	}

	public void setInd(List<Integer> ind) {
		Ind = ind;
	}

	public List<Double> getAlphas() {
		return alphas;
	}

	public void setAlphas(List<Double> alphas) {
		this.alphas = alphas;
	}
}
