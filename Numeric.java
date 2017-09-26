package vcu.edu.datastreamlearning.ollawv;

import com.yahoo.labs.samoa.instances.Instance;

public class Numeric {

	/**
	 * Calculates dot product between two instances
	 * @param	double[],double[]	instance and sv
	 * @return	double		dot product
	 */
	public static double dot(double[] x, double[] c, int dim){
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
	public static double norm2(Instance inst){
		double sums = 0.0;
		for(int j = 0; j < inst.numAttributes()-1; j++){
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
	public static void arrayMulConst(double val, int size, double[] array){
		for(int i = 0; i < size; i++){
			array[i] = array[i]*val;
		}
	}
	
	/**
	 * Sums two double arrays together and puts into result
	 * @param array
	 * @param result
	 */
	public static void arrayAdd(double[] array, int size, double[] result){
		for(int i = 0; i < size; i++){
			result[i] = result[i] + array[i];
		}
	}
	
	/**
	 * Adds a scalar to a double array and puts into result
	 * @param val
	 * @param result
	 */
	public static void arrayAddConst(double val, int size, double[] result){
		for(int i = 0; i < size; i++){
			result[i] = result[i] + val;
		}
	}

	/**
	 * Swap two elements in double array
	 * @param array
	 * @param i
	 * @param j
	 */
	public static void arraySwap(int i, int j, int[] array){
		int temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}

	/**
	 * Swap two elements in double array
	 * @param array
	 * @param i
	 * @param j
	 */
	public static void arraySwap(int i, int j, double[] array){
		double temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
}
