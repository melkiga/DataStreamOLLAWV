package vcu.edu.datastreamlearning.ollawv;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

public class Cache {
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
		xIndices.clear();
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
