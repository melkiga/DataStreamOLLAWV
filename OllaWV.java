/**
 * <!-- globalinfo-start --> 
 * Implements OLLA Worst Violator, written by Gabriella Melki & VK.
 * <!-- globalinfo-end -->
 */

package vcu.edu.datastreamlearning.ollawv;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.core.StringUtils;

public class OllaWV extends AbstractClassifier {
	private static final long serialVersionUID = 7225375378266360793L;

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public void trainOnInstances(Instances arg0) {
		// TODO Auto-generated method stub
		
	}

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

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		StringUtils.appendIndented(out, indent, toString());
        StringUtils.appendNewline(out);
	}

}
