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
	 * Seed value, default = 1
	 */
	public IntOption seedOption = new IntOption("seed",'r',"Seed value.",1);
	
	/**
	 * Data Header Variables
	 */
	public int num_data;
	public int dim;
	public int num_classes;
	public int standardize;
	/**
	 * Logger for debugging.
	 */
	private static Logger log = new Logger();
	/**
	 * Holds SVM Hyper-parameters.
	 */
	private static SVMParameters params = new SVMParameters();
	
	/**
	 * Sets options for model and initializes header for data.
	 */
	@Override
	public void setModelContext(InstancesHeader context){
		// Set SVM parameters from command-line
		params.setC(cOption.getValue());
		params.setGamma(gOption.getValue());
		params.setEpochs(eOption.getValue());
		params.setTol(tOption.getValue());
		// get data header
		this.num_data = context.numInstances();
		this.dim = context.numAttributes();
		this.num_classes = context.numClasses();
		this.standardize = standardizeOption.getValue();
		// if verbose, print out the model header
		if(vOption.getValue() == 1){
			log.printBarrier();
			log.print(this.getPurposeString());
			log.printBarrier();
			log.print(this.getModelContextString());
			log.printBarrier();
			log.printFormatted("Total number of data: %d\n", this.num_data);
			log.printFormatted("Dimensionality: %d\n", this.dim);
			log.printFormatted("Number of classes: %d\n", this.num_classes);
			log.printFormatted("Standardize data: %d\n", this.standardize);
			log.printFormatted("Seed: %d\n", this.randomSeed);
			log.printBarrier();
		}
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
		log.print("ERROR: trainOnInstanceImpl was called and no implementation for this yet.\n");
		System.err.println("trainOnInstanceImpl::This sould not happen.");
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
        return "OnLine Learning Algorithm using Worst Violator (OLLAWV), VK and GM (2017).\n";
    }
	
	/**
	 * Prints the model context.
	 */
	public String getModelContextString(){
		StringBuffer buff = new StringBuffer();
		buff.append("Data Standardization: "+standardize+"\n");
		buff.append("SVM Parameter C: "+params.getC()+"\n");
		buff.append("RBF Parameter gamma: "+params.getGamma()+"\n");
		buff.append("Using Bias: "+params.getBetta()+"\n");
		buff.append("Pruning tolerance: "+params.getTol()+"\n");
		buff.append("Number of Epochs: "+params.getEpochs()+"\n");
		buff.append("Using Change Detection: "+changeOption.getValue()+"\n");
        return buff.toString();
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
