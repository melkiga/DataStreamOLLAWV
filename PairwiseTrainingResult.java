package vcu.edu.datastreamlearning.ollawv;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
/**
 * This is equivalent to the PairwiseTrainingResult
 * @author Gabriella Melki
 *
 */
public class PairwiseTrainingResult {
	/**
	 * Alpha values
	 */
	private List<Double> alphas;
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
	 * Data class pair
	 */
	Tuple<Integer,Integer> trainingLabels;
	/**
	 * Support vector data
	 */
	private int[] samples;
	
	/**
	 * Constructor for OLLAWV model
	 */
	public PairwiseTrainingResult(SVMParameters params, int size, Tuple<Integer,Integer> trainingLabels){
		this.setParams(params);
		setAlphas(new ArrayList<Double>(size));
		setSvnumber(0);
		setBias(0.0);
		this.trainingLabels = trainingLabels;
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

	public List<Double> getAlphas() {
		return alphas;
	}

	public void setAlphas(List<Double> alphas) {
		this.alphas = new ArrayList<Double>(alphas);
	}
	
	public double getAlpha(int loc){
		return this.alphas.get(loc);
	}

	public int[] getSamples() {
		return samples;
	}

	public void setSamples(int[] samples) {
		this.samples = Arrays.copyOf(samples, svnumber);
	}
	
	public void setSample(int loc, int val){
		this.samples[loc] = val;
	}
	
	public int getSample(int loc){
		return this.samples[loc];
	}
}
