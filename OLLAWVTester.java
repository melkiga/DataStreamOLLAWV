package vcu.edu.datastreamlearning.ollawv;

import moa.DoTask;


public class OLLAWVTester {
	
	public static void main(String[] args){
		
		Logger log = new Logger();
		/* Set up data configuration. */
		int numChunks = 2;
		int chunkSize = 20;
		int dim = 2;
		int seed = 1;
		int numClasses = 2;
		String[] localArgs = {"moa.tasks.EvaluateInterleavedChunks -S ", 
				"-s \"generators.RandomRBFGenerator -r "+seed+" -i "+seed+" -c "+numClasses+" -a "+dim+"\"", 
				"-l \"vcu.edu.datastreamlearning.ollawv.OLLASolver -v 1\"", 
				"-i "+chunkSize*numChunks, 
				"-f "+dim, 
				"-c "+chunkSize};
		
		/* Print configuration. */
		log.printBarrier();
		log.printFormatted("Number of chunks: %d\n", numChunks);
		log.printFormatted("Number of instances: %d\n", chunkSize);
		log.printFormatted("Dimensionality: %d\n", dim);
		log.printFormatted("Number of classes: %d\n", numClasses);
		log.printFormatted("Seed: %d\n", seed);
		log.printBarrier();
		
		/* Calls Evaluate Interleaved Chunks */
		DoTask.main(localArgs);
	}

}
