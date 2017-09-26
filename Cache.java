package vcu.edu.datastreamlearning.ollawv;

import java.util.ArrayList;
import java.util.List;

import com.yahoo.labs.samoa.instances.Instances;

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
	/**
	 * Kernel parameters
	 */
	SVMParameters params;
	List<double[]> xSV;
	List<Double> ySV;
	double bias;
	double yyNeg;
	/**
	 * Kernel evaluator
	 */
	KernelEvaluator eval;
	
	public Cache(Instances data, int probSize, SVMParameters par){
		// cache variables
		problemSize = probSize;
		currentSize = probSize;
		params = par;
		svnumber = 1;
		eval = new KernelEvaluator(data, probSize, par.getGamma());
		
		initialize(data);
	}
	
	/**
	 * Initialize (fill) the cache + kernel evaluator variables
	 */
	protected void initialize(Instances data){
		// cache variables
		setAlphas(new ArrayList<Double>());
		output = new double[problemSize];
		// evaluator variables
		xSV = new ArrayList<double []>();
		ySV = new ArrayList<Double>();
		bias = 0.0;
		yyNeg = -1.0;
	}
	
	/**
	 * Training a binary OLLAWV model
	 */
	protected void trainForCache(){
		int maxIter = (int) Math.ceil(params.getEpochs()*currentSize);
		double C = params.getC();
		double margin = params.getTol()*C;
		double eta = 0.0;
		double lambda = 0.0;
		double LB = 0.0;
		double betta = params.getBetta(); 
		
		// loop variables
		int iter = 0;
		Violator viol = new Violator(iter,0.0);
		double[] G = new double[currentSize];
		//Instance sample;
		double label;
		
		do {
			iter += 1.0;
			eta = 2.0 / Math.sqrt(iter);
			
			// get the sample
			//sample = data.get(viol.violator);
			label = getLabel(viol.violator); 
			
			// set update variables
			lambda = eta*C*label;
			LB = (lambda*betta) / currentSize;
			
			// calculate the kernel vector
			eval.evalKernel(viol.violator,currentSize,G);
			
			// calculate the output vector (output = output + G*lambda + LB)
			Numeric.arrayMulConst(lambda,currentSize,G); 
			Numeric.arrayAdd(G,currentSize,output);
			Numeric.arrayAddConst(LB,currentSize,output);
			
			// update worst violator information
			alphas.add(lambda);
			bias = bias + LB;
			ySV.add(label);
			
			// find worst violator
			viol = findWorstViolator();
			
			// perform sv update
			viol.violator = performSVUpdate(viol.violator);
			
		} while(viol.yo < margin && iter < maxIter); 
	}
	
	/**
	 * Finds index and value of worst violator excluding support vectors
	 * @param data
	 * @return
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
		if(v >= svnumber){
			eval.swap(v, svnumber); 
			Numeric.arraySwap(v, svnumber, output); // output
			
			v = svnumber;
			svnumber++;
		}
		return v;
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
		xSV.clear();
		ySV.clear();
		svnumber = 1;
	}
	
	/**
	 * Gets the label of sample v
	 */
	public double getLabel(int u){
		double[] vals = {-1.0, 1.0};
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
