/**
 * <!-- globalinfo-start --> 
 * Implements OLLA Worst Violator Solver, written by Gabriella Melki & VK.
 * <!-- globalinfo-end -->
 */

package vcu.edu.datastreamlearning.ollawv;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.core.StringUtils;

public class OLLASolver extends AbstractClassifier {
	private static final long serialVersionUID = 7225375378266360793L;
	
	/**
	 * Logger for debugging.
	 */
	private static Logger log;
	
	/**
	 * SVM Penalty Parameter C, command line, default value = 1.0
	 */
	public FloatOption cOption = new FloatOption("C", 'c', "SVM Penalty Parameter.", 1.0);
	/**
	 * Gaussian Gamma Parameter, command line, default value = 0.5
	 */
	public FloatOption gOption = new FloatOption("gamma", 'g', "Gaussian RBF Gamma.", 0.5);
	/**
	 * Binary value to choose whether to use the bias term or not, command line, default value = 1
	 */
	public IntOption bOption = new IntOption("betta",'b',"Use bias or not.",1);
	/**
	 * Set the number of epochs, command line, default value = 0.5
	 */
	public FloatOption eOption = new FloatOption("epochs",'e',"Number of epochs.",0.5);
	/**
	 * Whether to use change-detection method, command line, default value = 0
	 */
	public IntOption changeOption = new IntOption("detection",'d',"Use change detection.",0);
	/**
	 * Set whether to print the entire model, default value = 0
	 */
	public IntOption vOption = new IntOption("verbose",'v',"Print entire model.",0);
	/**
	 * Option for margin width, default value 0.1
	 */
	public FloatOption tOption = new FloatOption("tol",'t',"Tolerance for alpha pruning.",0.1);
	/**
	 * Option to standardize data, default = 1
	 */
	public IntOption standardizeOption = new IntOption("standardize",'p',"Standardize data, 0 mean 1 stdv",0);
	
	/**
	 * Sets options for model.
	 */
	@Override
	public void setModelContext(InstancesHeader context){
		SVMParameters params = new SVMParameters();
		params.setC(cOption.getValue());
		params.setGamma(gOption.getValue());
		params.setEpochs(eOption.getValue());
		params.setTol(tOption.getValue());
	}
	
	@Override
	public boolean isRandomizable() {
		return false;
	}
	
	/**
	 * Train on a batch of instances.
	 */
	@Override
	public void trainOnInstances(Instances arg0) {
		// TODO Auto-generated method stub
		
	}
	
	/** 
	 * Test on a batch of instances.
	 */
	@Override
	public double[] getVotesForInstance(Instance inst) {
		// TODO Tester method for single instance
		return null;
	}
	
	// TODO add toString method
	// TODO add reset method

	@Override
	public void resetLearningImpl() {
		// TODO reset the learner
		
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// TODO Trains the classifier on a given instance
		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}
	
	/**
	 *  Returns the name of the solver. 
	 */
	@Override
    public String getPurposeString() {
        return "OnLine Learning Algorithm using Worst Violator (OLLAWV), VK and GM (2017).";
    }
	
	/**
	 *  Prints parameter out to string builder (auto-generated inherited). 
	 */
	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		StringUtils.appendIndented(out, indent, toString());
        StringUtils.appendNewline(out);
	}

	public static Logger getLog() {
		return log;
	}

	public static void setLog(Logger log) {
		OLLASolver.log = log;
	}

}
