package vcu.edu.datastreamlearning.ollawv;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

public class Numeric {
	
	protected Instances data;
	protected double[] x2;
	protected int problemSize;
	
	protected double gamma;
	
	public Numeric(){
		
	}
	
	public Numeric(Instances data, int problemSize){
		this.problemSize = problemSize;
		x2 = new double[problemSize];
		
		for(int i = 0; i < problemSize; i++){
			x2[i] = norm2(data.get(i));
		}
	}
	
	/**
	 * Calculates RBF Gaussian kernel vector of indwviol
	 * @param data
	 * @param indwviol
	 * @param G
	 */
	public void evalKernel(Instances data, int indwviol, int to, double[] G){
		evalDist(data, indwviol, to, G);
		
		for(int i = 0; i < to; i++){
			G[i] = Math.exp(-G[i]*gamma);
		}
	}
	
	/**
	 * Evaluates Euclidean distance between indwviol and the rest of the data
	 * @param data
	 * @param indwviol
	 * @param G
	 */
	public void evalDist(Instances data, int indwviol, int to, double[] G){
		double result = 0.0;
		int dim = data.get(0).numAttributes()-1;
		for(int i = 0; i < to; i++){
			double x2_id = x2[i];
			double x2_i = x2[indwviol];
			result = dot(data.get(i).toDoubleArray(), data.get(indwviol).toDoubleArray(), dim);
			G[i] = x2_id + x2_i -2*result;
		}
	}
	
	/**
	 * Calculates dot product between two instances
	 * @param	double[],double[]	instance and sv
	 * @return	double		dot product
	 */
	public double dot(double[] x, double[] c, int dim){
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
		for(int j = 0; j < inst.numAttributes()-1; j++){
			sums += Math.pow(inst.value(j), 2);
		}
		return sums;
	}
}
