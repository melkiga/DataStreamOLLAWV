package vcu.edu.datastreamlearning.ollawv;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.yahoo.labs.samoa.instances.Instance;
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
	private List<double[]> xSV;
	private List<Double> ySV;
	private int[] samples;
	private List<Double> x2;
	
	/**
	 * Constructor for OLLAWV model
	 */
	public PairwiseTrainingResult(SVMParameters params, int size, Tuple<Integer,Integer> trainingLabels){
		this.setParams(params);
		setAlphas(new ArrayList<Double>(size));
		setxSV(new ArrayList<double[]>(size));
		setySV(new ArrayList<Double>(size));
		setX2(new ArrayList<Double>(size));
		setSvnumber(0);
		setBias(0.0);
		this.trainingLabels = trainingLabels;
	}
	
	/**
	 * Calculates RBF Gaussian kernel vector of xSV and instance. This is called by testing function.
	 * @param inst - unknown test instance
	 * @param x - X data for support vectors
	 * @param x2 - Norm squared for support vectors
	 * @param svnum - Max number of SV
	 * @param G - Kernel vector
	 */
	public void evalInnerKernel(Instance inst, List<double[]> x, List<Double> x2, double[] G){
		// evaluate distance
		double result = 0.0;
		double x2_i = Numeric.norm2(inst);
		int dim = inst.numAttributes()-1;
		int size = x.size();
		for(int i = 0; i < size; i++){
			double x2_id = x2.get(i);
			result = Numeric.dot(inst.toDoubleArray(), x.get(i), dim);
			G[i] = x2_id + x2_i -2*result;
		}
		
		for(int i = 0; i < size; i++){
			G[i] = Math.exp(-G[i]*params.getGamma());
		}
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

	public List<double[]> getxSV() {
		return xSV;
	}

	public void setxSV(List<double[]> xSV) {
		this.xSV = new ArrayList<double[]>(xSV);
	}

	public List<Double> getySV() {
		return ySV;
	}

	public void setySV(List<Double> ySV) {
		this.ySV = new ArrayList<Double>(ySV);
	}

	public int[] getSamples() {
		return samples;
	}

	public void setSamples(int[] samples) {
		this.samples = Arrays.copyOf(samples, svnumber);
	}

	public List<Double> getX2() {
		return x2;
	}

	public void setX2(List<Double> x2) {
		this.x2 = new ArrayList<Double>(x2);
	}
}
