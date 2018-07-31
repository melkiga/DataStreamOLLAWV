package vcu.edu.datastreamlearning.ollawv;

import java.util.ArrayList;
import java.util.List;

public class Cache {
	private static final int INT_MAX = 999999999;
	/**
	 * Violator, contains the output value and index of a violator
	 */
	protected class Violator {
		int violator;
		double yo;
		
		Violator(int violator, double yo){
			this.violator = violator;
			this.yo = yo;
		}
	}
	/**
	 * Alpha values
	 */
	private List<Double> alphas;
	/**
	 * Cache parameters
	 */
	int problemSize;
	int currentSize;
	int svnumber;
	double[] output;
	int[] forwardOrder;
	int[] backwardOrder;
	SVMParameters params;
	double bias;
	double yyNeg;
	/**
	 * Kernel evaluator
	 */
	KernelEvaluator eval;
	double[] vals = {-1.0, 1.0};
	
	public Cache(int probSize, SVMParameters par, KernelEvaluator eval){
		// cache variables
		problemSize = probSize;
		currentSize = probSize;
		params = par;
		svnumber = 1;
		this.eval = eval;
		
		initialize();
	}
	
	/**
	 * Initialize (fill) the cache + kernel evaluator variables
	 */
	protected void initialize(){
		// cache variables
		setAlphas(new ArrayList<Double>());
		output = new double[problemSize];
		bias = 0.0;
		yyNeg = -1.0;
		// forward and backward order
		forwardOrder = new int[problemSize];
		backwardOrder = new int[problemSize];
		for(int i = 0; i < problemSize; i++){
			forwardOrder[i] = i;
			backwardOrder[i] = i;
		}
	}
	
	/**
	 * Training a binary OLLAWV model
	 */
	protected void trainForCache(){
		double margin = params.getC();
		double eta = 0.0;
		double lambda = 0.0;
		double LB = 0.0;
		double betta = params.getBetta(); 
		double label;
		
		// loop variables
		int iter = 0;
		Violator viol = new Violator(iter,0.0);
		double[] G = new double[currentSize];
		
		do {
			iter += 1.0;
			eta = 2.0 / Math.sqrt(iter);
			
			// get the sample label
			label = getLabel(viol.violator); 
			
			// set update variables
			lambda = eta*label;
			LB = (lambda*betta) / currentSize;
			
			// calculate the kernel vector
			eval.evalKernel(viol.violator,svnumber,currentSize,G);
			
			// calculate the output vector (output = output + G*lambda + LB)
			updateOutput(lambda, G, LB);
			
			// update worst violator information
			alphas.add(lambda);
			bias = bias + LB;
			
			// find worst violator
			viol = findWorstViolator();
			
			// perform SV update
			viol.violator = performSVUpdate(viol.violator);
			
		} while(viol.yo < margin && iter < currentSize-1); 
	}
	
	/**
	 * Finds index and value of worst violator excluding support vectors
	 * @param data
	 * @return Worst Violator
	 */
	public Violator findWorstViolator(){
		double min_val = INT_MAX;
		double ksi = 0.0;
		double label = 0.0;
		Violator worstViol = new Violator(svnumber,0);
		for(int i = svnumber; i < currentSize; i++){
			label = getLabel(i);
			ksi = output[i] * label;
			if(ksi < min_val){
				worstViol.violator = i;
				worstViol.yo = ksi;
				min_val = ksi;
			}
		}
		return worstViol;
	}
	
	/**
	 * Swaps the samples to stack the support vectors on top
	 * @param v
	 */
	public int performSVUpdate(int v){
		this.swapSamples(v, svnumber);
		
		v = svnumber;
		svnumber++;
		return v;
	}
	
	/**
	 * (output = output + G*lambda + LB)
	 * @param lambda
	 * @param G
	 * @param LB
	 */
	public void updateOutput(double lambda, double[] G, double LB){
		for(int i = svnumber; i < currentSize; i++){
			output[i] = output[i] + G[i]*lambda + LB;
		}
	}
	
	/**
	 * Swaps samples + their attributes
	 * @param u
	 * @param v
	 */
	public void swapSamples(int u, int v){
		eval.swap(u, v);
		Numeric.arraySwap(u, v, output);
		
		forwardOrder[backwardOrder[u]] = v;
		forwardOrder[backwardOrder[v]] = u;
		Numeric.arraySwap(u, v, backwardOrder);
	}
	
	/**
	 * Clear the cache
	 */
	public void reset(){
		alphas.clear();
		for(int i = 0; i < currentSize; i++){
			output[i] = 0.0;
		}
		bias = 0.0;
		svnumber = 1;
	}
	
	/**
	 * Gets the label of sample v
	 */
	public double getLabel(int u){
		double lab = eval.getLabel(u);
		int index = (lab == yyNeg) ? 1 : 0;
		return vals[index];
	}
	
	/**
	 * Sets the negative label
	 */
	public void setLabel(int lab){
		yyNeg = lab;
	}
	
	/**
	 * Sets current size
	 */
	public void setCurrentSize(int size){
		currentSize = size;
		eval.setCurrentSize(size);
	}
	
	public List<Double> getAlphas() {
		return alphas;
	}
	public void setAlphas(List<Double> alphas) {
		this.alphas = alphas;
	}
}
