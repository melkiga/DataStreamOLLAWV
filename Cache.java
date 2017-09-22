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
	double[] output;
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
		output = new double[problemSize];
		// evaluator variables
		x2 = new ArrayList<Double>();
		xSV = new ArrayList<double []>();
		ySV = new ArrayList<Double>();
		bias = 0.0;
		yyNeg = -1.0;
		
		for(int i = 0; i < problemSize; i++){
			x2.add(norm2(data.get(i)));
			output[i] = 0.0;
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
		int indwviol = xIndices.get(0);
		Violator viol = new Violator(indwviol,0.0);
		double[] G = new double[currentSize];
		Instance sample;
		double label = data.get(xIndices.get(indwviol)).classValue();
		
		while(iter < maxIter && viol.yo < margin){
			iter += 1.0;
			eta = 2.0 / Math.sqrt(iter);
			// set update variables
			lambda = eta*C*getLabel(viol.violator, label);
			LB = (lambda*betta) / currentSize;
			
			// get the sample
			sample = data.get(viol.violator);
			// calculate the kernel vector
			evalKernel(sample,viol.violator,G);
			// calculate the output vector (output = output + G*lambda + LB)
			arrayMulConst(lambda,G); 
			arrayAdd(G,output);
			arrayAddConst(LB,output);
			// update worst violator information
			alphas.set(viol.violator, alphas.get(viol.violator) + lambda);
			bias = bias + LB;
			Ind.add(viol.violator);
			xSV.add(sample.toDoubleArray());
			ySV.add(label);
			svnumber += 1;

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
	 * Calculates the RBF kernel vector of instance id, in the range [from,to), based on
	 * the number of support vectors, and returns the vector in G.
	 * @param id	- the instance
	 * @param from	- starting svec
	 * @param to	- ending svec
	 * @param G		- Gaussian kernel vector of instance id
	 */
	public void evalKernel(Instance sample, int id, double[] G){
		evalDist(sample, id, G);
		
		for(int i = 0; i < svnumber; i++){
			G[i] = Math.exp(-G[i]*params.getGamma());
		}
	}
	
	/**
	 * Evaluates the squared Euclidean distance of a training sample
	 * @param id
	 * @param to
	 * @param G
	 */
	public void evalDist(Instance sample, int id, double[] G){
		double result = 0.0;
		for(int i = 0; i < svnumber; i++){
			double x2_id = x2.get(id);
			double x2_i = x2.get(Ind.get(i));
			result = dot(sample.toDoubleArray(), xSV.get(i));
			G[i] = x2_id + x2_i -2*result;
		}
	}
	
	/**
	 * Calculates dot product between two instances
	 * @param	double[],double[]	instance and sv
	 * @return	double		dot product
	 */
	public double dot(double[] x, double[] c){
		double sums = 0.0;
		for(int j = 0; j < dim; j++){
			sums += x[j]*c[j];
		}
		return sums;
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
	
	/**
	 * Scales double array by val
	 * @param array
	 * @param val
	 * @return scaledArray
	 */
	public void arrayMulConst(double val, double[] array){
		for(int i = 0; i < currentSize; i++){
			array[i] = array[i]*val;
		}
	}
	
	/**
	 * Sums two double arrays together and puts into result
	 * @param array
	 * @param result
	 */
	public void arrayAdd(double[] array, double[] result){
		for(int i = 0; i < currentSize; i++){
			result[i] = result[i] + array[i];
		}
	}
	
	/**
	 * Adds a scalar to a double array and puts into result
	 * @param val
	 * @param result
	 */
	public void arrayAddConst(double val, double[] result){
		
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
