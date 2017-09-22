package vcu.edu.datastreamlearning.ollawv;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

public class Cache {
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
	 * Support vector indices
	 */
	private List<Integer> Ind;
	/**
	 * Cache parameters
	 */
	int problemSize;
	int currentSize;
	int svnumber;
	/**
	 * Kernel parameters
	 */
	SVMParameters params;
	int dim;
	List<Double> x2;
	List<double[]> xSV;
	List<Double> ySV;
	double bias;
	List<Integer> xIndices;
	/**
	 * Classes that have value 0, take the negative class, otherwise they are positive
	 */
	double[] yVal = {-1.0, 1.0};
	double yyNeg;
	
	public Cache(Instances data, int probSize, int d, SVMParameters par){
		// cache variables
		problemSize = probSize;
		currentSize = probSize;
		dim = d;
		params = par;
		svnumber = 1;
		
		initialize(data);
	}
	
	/**
	 * Initialize (fill) the cache + kernel evaluator variables
	 */
	protected void initialize(Instances data){
		// cache variables
		setAlphas(new ArrayList<Double>(Collections.nCopies(svnumber, 0.0)));
		setInd(new ArrayList<Integer>());
		// evaluator variables
		x2 = new ArrayList<Double>();
		xSV = new ArrayList<double []>();
		ySV = new ArrayList<Double>();
		bias = 0.0;
		yyNeg = -1.0;
		
		for(int i = 0; i < problemSize; i++){
			x2.add(norm2(data.get(i)));
		}
	}
	
	/**
	 * Training a binary OLLAWV model
	 */
	protected void trainForCache(Instances data){
		int maxIter = (int) Math.ceil(params.getEpochs()*currentSize);
		double C = params.getC();
		double margin = params.getTol()*C;
		double eta = 0.0;
		double lambda = 0.0;
		double LB = 0.0;
		double betta = params.getBetta(); 
		
		// loop variables
		int iter = 0;
		int indCounter = 0;
		int ind = xIndices.get(indCounter);
		Violator viol = new Violator(ind,0.0);
		double[] output = new double[currentSize];
		double[] G = new double[currentSize];
		Instance sample;
		double label = data.get(xIndices.get(ind)).classValue();
		
		while(iter < maxIter && viol.yo < margin){
			iter += 1.0;
			eta = 2.0 / Math.sqrt(iter);
			// get the sample
			sample = data.get(ind);
			
			
			lambda = eta*C*getLabel(viol.violator, label);
			LB = (lambda*betta) / currentSize;
			// TODO: perform update
			// TODO: find worst violator
			// TODO: perform sv update
			
		}
		
	}
	
	/**
	 * Gets the label of sample v
	 */
	public double getLabel(int v, double lab){
		double[] vals = {1.0, -1.0};
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
	}
	
	/**
	 * Sets the indices of the data to use for training the model
	 */
	protected void setIndices(List<Integer> first, List<Integer> second){
		xIndices = new ArrayList<Integer>(first);
		xIndices.addAll(second);
	}
	
	/**
	 * Calculates the Euclidean Norm Squared of an instance
	 * @param	Instance 
	 * @return	double = sum(x_i^2)
	 */
	public double norm2(Instance inst){
		double sums = 0.0;
		for(int j = 0; j < dim; j++){
			sums += Math.pow(inst.value(j), 2);
		}
		return sums;
	}
	
	public List<Double> getAlphas() {
		return alphas;
	}
	public void setAlphas(List<Double> alphas) {
		this.alphas = alphas;
	}
	public List<Integer> getInd() {
		return Ind;
	}
	public void setInd(List<Integer> ind) {
		Ind = ind;
	}
}
