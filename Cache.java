package vcu.edu.datastreamlearning.ollawv;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.yahoo.labs.samoa.instances.Instance;
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
		setAlphas(new ArrayList<Double>());
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
		Violator viol = new Violator(iter,0.0);
		double[] G = new double[currentSize];
		Instance sample;
		double label;
		
		do {
			iter += 1.0;
			eta = 2.0 / Math.sqrt(iter);
			
			// get the sample
			sample = data.get(viol.violator);
			label = getLabel(data.get(viol.violator).classValue()); 
			
			// set update variables
			lambda = eta*C*label;
			LB = (lambda*betta) / currentSize;
			
			// calculate the kernel vector
			evalKernel(data,viol.violator,G);
			
			// calculate the output vector (output = output + G*lambda + LB)
			arrayMulConst(lambda,G); 
			arrayAdd(G,output);
			arrayAddConst(LB,output);
			
			// update worst violator information
			alphas.add(lambda);
			bias = bias + LB;
			Ind.add(viol.violator);
			xSV.add(sample.toDoubleArray());
			ySV.add(label);

			// find worst violator
			viol = findWorstViolator(data);
			
			// perform sv update
			viol.violator = performSVUpdate(data,viol.violator);
			
		} while(iter < maxIter && viol.yo < margin);
	}
	
	/**
	 * Finds index and value of worst violator excluding support vectors
	 * @param data
	 * @return
	 */
	public Violator findWorstViolator(Instances data){
		double min_val = INT_MAX;
		double ksi = 0.0;
		double label = 0.0;
		Violator worstViol = new Violator(svnumber,0);
		for(int i = svnumber; i < currentSize; i++){
			label = getLabel(data.get(i).classValue());
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
	public int performSVUpdate(Instances data, int v){
		if(v >= svnumber){
			data.swap(v, svnumber); // data
			Collections.swap(x2, v, svnumber); // x2
			arraySwap(output, v, svnumber); // output
			
			v = svnumber;
			svnumber += 1;
		}
		return v;
	}
	
	/**
	 * Calculates RBF Gaussian kernel vector of indwviol
	 * @param data
	 * @param indwviol
	 * @param G
	 */
	public void evalKernel(Instances data, int indwviol, double[] G){
		evalDist(data, indwviol, G);
		
		for(int i = 0; i < currentSize; i++){
			G[i] = Math.exp(-G[i]*params.getGamma());
		}
	}
	
	/**
	 * Evaluates Euclidean distance between indwviol and the rest of the data
	 * @param data
	 * @param indwviol
	 * @param G
	 */
	public void evalDist(Instances data, int indwviol, double[] G){
		double result = 0.0;
		for(int i = 0; i < currentSize; i++){
			double x2_id = x2.get(i);
			double x2_i = x2.get(indwviol);
			result = dot(data.get(i).toDoubleArray(), data.get(indwviol).toDoubleArray());
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
		for(int i = 0; i < currentSize; i++){
			result[i] = result[i] + val;
		}
	}
	
	/**
	 * Swap two elements in double array
	 * @param array
	 * @param i
	 * @param j
	 */
	public void arraySwap(double[] array, int i, int j){
		double temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
	
	/**
	 * Gets the label of sample v
	 */
	public double getLabel(double lab){
		double[] vals = {-1.0, 1.0};
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
