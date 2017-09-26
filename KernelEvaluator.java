package vcu.edu.datastreamlearning.ollawv;

import com.yahoo.labs.samoa.instances.Instances;

public class KernelEvaluator {
	
	protected Instances data;
	protected int problemSize;
	protected double[] x2;
	int[] samples;
	
	double bias;
	
	protected double gamma;
	
	/**
	 * Constructor
	 * @param data
	 * @param problemSize
	 */
	public KernelEvaluator(Instances data, int problemSize, double gamma){
		this.data = data;
		this.problemSize = problemSize;
		this.gamma = -gamma;
		this.bias = 0.0;
		
		// initialize x^2
		x2 = new double[problemSize];
		samples = new int[problemSize];
		for(int i = 0; i < problemSize; i++){
			x2[i] = Numeric.norm2(data.get(i));
			samples[i] = i;
		}
	}
	
	/**
	 * Calculates RBF Gaussian kernel vector of indwviol
	 * @param data
	 * @param indwviol
	 * @param G
	 */
	public void evalKernel(int indwviol, int to, double[] G){
		evalDist(indwviol, to, G);
		
		for(int i = 0; i < to; i++){
			G[i] = Math.exp(G[i]*gamma);
		}
	}
	
	/**
	 * Evaluates Euclidean distance between indwviol and the rest of the data
	 * @param data
	 * @param indwviol
	 * @param G
	 */
	public void evalDist(int indwviol, int to, double[] G){
		double result = 0.0;
		int dim = data.get(0).numAttributes()-1;
		for(int i = 0; i < to; i++){
			double x2_id = x2[i];
			double x2_i = x2[indwviol];
			result = Numeric.dot(data.get(i).toDoubleArray(), data.get(indwviol).toDoubleArray(), dim);
			G[i] = x2_id + x2_i -2*result;
		}
	}
	
	/**
	 * Swaps values of members
	 * @param i
	 * @param j
	 */
	public void swap(int i, int j){
		data.swap(i, j); 
		Numeric.arraySwap(i, j, x2);
		Numeric.arraySwap(i, j, samples);
	}
	
	public void reset(){
		for(int i = 0; i < problemSize; i++){
			samples[i] = i;
		}
	}
	
	/**
	 * Gets the class label
	 * @param u
	 * @return
	 */
	public double getLabel(int u){
		return data.get(u).classValue();
	}

	public void setCurrentSize(int size) {
		this.problemSize = size;
	}
}
