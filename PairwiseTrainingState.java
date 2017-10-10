package vcu.edu.datastreamlearning.ollawv;

import java.util.List;

/**
 * Pairwise training state holds multiple pairwise OLLAWV models.
 * @author gabriellamelki
 *
 */
public class PairwiseTrainingState {
	/**
	 * List of models
	 */
	protected List<PairwiseTrainingResult> models;
	/**
	 * Max number of support vectors for all models
	 */
	private int svNumber;
	/**
	 * Max number of labels
	 */
	private int labelNumber;
	/**
	 * Testing holders for models
	 */
	protected double[] votes;
	protected double[] evidence;
	
	public PairwiseTrainingState(){
		models = null;
		svNumber = -1;
		labelNumber = -1;
		setVotes(null);
		setEvidence(null);
	}
	
	public List<PairwiseTrainingResult> getModels() {
		return models;
	}
	public void setModels(List<PairwiseTrainingResult> models) {
		this.models = models;
	}
	public int getSvNumber() {
		return svNumber;
	}
	public void setSvNumber(int svNumber) {
		this.svNumber = svNumber;
	}
	public int getLabelNumber() {
		return labelNumber;
	}
	public void setLabelNumber(int labelNumber) {
		this.labelNumber = labelNumber;
	}

	public double[] getVotes() {
		return votes;
	}

	public void setVotes(double[] votes) {
		this.votes = votes;
	}

	public double[] getEvidence() {
		return evidence;
	}

	public void setEvidence(double[] evidence) {
		this.evidence = evidence;
	}
}
