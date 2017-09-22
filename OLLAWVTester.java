package vcu.edu.datastreamlearning.ollawv;

import moa.DoTask;


public class OLLAWVTester {
	
	public static void main(String[] args){
		
		Logger log = new Logger();
		/* Set up data configuration. */
		int numChunks = 2;
		int chunkSize = 50;
		int dim = 2;
		int seed = 1;
		int numClasses = 3;
		String[] localArgs = {"moa.tasks.EvaluateInterleavedChunks -S ", 
				"-s \"generators.RandomRBFGenerator -n 3 -r "+seed+" -i "+seed+" -c "+numClasses+" -a "+dim+"\"", 
				"-l \"vcu.edu.datastreamlearning.ollawv.OLLASolver -v 1\"", 
				"-i "+chunkSize*numChunks, 
				"-f "+dim, 
				"-c "+chunkSize};
		
		/* Print configuration. */
		log.printBarrier();
		log.printf("Number of chunks: %d\n", numChunks);
		log.printf("Number of instances: %d\n", chunkSize);
		log.printf("Dimensionality: %d\n", dim);
		log.printf("Number of classes: %d\n", numClasses);
		log.printf("Seed: %d\n", seed);
		log.printBarrier();
		
		/* Calls Evaluate Interleaved Chunks */
		DoTask.main(localArgs);
		
		log.close();
	}

}
