package perceptron;

import java.io.File;
import java.util.ArrayList;



import java.util.Arrays;
import java.util.List;
import edu.stanford.nlp.ling.CoreLabel;
import gnu.trove.TIntArrayList;
import utils.*;



public class Main {

	public static void main(String[] args) throws Exception {
		BiocXmlParser xmlParser = new BiocXmlParser(new File("data/Bioc.dtd").getAbsolutePath());
		ArrayList<BiocDocument> documents = xmlParser.parseBiocXmlFile(new File("data/CDR_sample.xml").getAbsolutePath());
		Tool tool = new Tool();
		tool.sentSplit = new SentenceSplitter(new Character[]{';'}, false, "data/common_english_abbr.txt");

		// Please create the directory before you run this function. 
		String instance_dir = "E:/perceptron/instance";
		preprocess(instance_dir, documents, tool);
		
		
		// Prepare training data
		List<BiocDocument> trainDocuments = new ArrayList<>(documents);
		ArrayList<PerceptronInputData> trainInputDatas = new ArrayList<PerceptronInputData>();
		ArrayList<PerceptronOutputData> trainOutputDatas = new ArrayList<PerceptronOutputData>();
		for(int j=0;j<trainDocuments.size();j++) {
			BiocDocument document = trainDocuments.get(j);
			List<File> files = Arrays.asList(new File(instance_dir+"/"+document.id).listFiles());
			for(File file:files) {
				if(file.getName().indexOf("input") != -1) {
					trainInputDatas.add((PerceptronInputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
				} else {
					trainOutputDatas.add((PerceptronOutputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
				}
			}
		}
		
		// Create a perceptron
		ArrayList<String> alphabetEntityType = new ArrayList<String>(Arrays.asList("Disease","Chemical"));
		int beamSize = 2;
		TIntArrayList d = new TIntArrayList();
		d.add(5);
		d.add(5);
		
		Perceptron perceptron = new Perceptron(alphabetEntityType, d);
		
		ArrayList<PerceptronFeatureFunction> featureFunctions = new ArrayList<PerceptronFeatureFunction>(
				Arrays.asList(new EntityFeatures(perceptron)));
		perceptron.setFeatureFunction(featureFunctions);
		perceptron.buildFeatureAlphabet(trainInputDatas, trainOutputDatas, tool);
		
		// train
		System.out.println("begin to train with "+perceptron.featureAlphabet.size()+" features");
		perceptron.trainPerceptron(10, beamSize, trainInputDatas, trainOutputDatas, tool);
		
		// test
		int countPredictEntity = 0;
		int countTrueEntity = 0;
		int countCorrectEntity = 0;
		List<BiocDocument> testDocuments = new ArrayList<>(documents);
		for(int j=0;j<testDocuments.size();j++) { // for each test file
			// prepare input
			BiocDocument document = testDocuments.get(j);
			ArrayList<PerceptronInputData> inputDatas = new ArrayList<PerceptronInputData>();
			List<File> files = Arrays.asList(new File(instance_dir+"/"+document.id).listFiles());
			for(File file:files) {
				if(file.getName().indexOf("input") != -1) {
					inputDatas.add((PerceptronInputData)ObjectSerializer.readObjectFromFile(file.getAbsolutePath()));
				} 
			}
			
			// predict
			ArrayList<Entity> preEntities = new ArrayList<Entity>();
			for(PerceptronInputData inputdata:inputDatas) {
				PerceptronStatus returnType = perceptron.beamSearch((PerceptronInputData)inputdata, null, false, beamSize, tool);
				PerceptronOutputData output = (PerceptronOutputData)returnType.z;
				
				for(int k=0;k<output.segments.size();k++) {
					Entity segment = output.segments.get(k);
					if(segment.type.equals("Disease") || segment.type.equals("Chemical"))
						preEntities.add(segment);
				}
			}
			
			countPredictEntity+= preEntities.size();
			ArrayList<Entity> goldEntities = document.entities;
			countTrueEntity += goldEntities.size();

			for(Entity preEntity:preEntities) {
				for(Entity goldEntity:goldEntities) {
					if(preEntity.equals(goldEntity)) {
						countCorrectEntity++;
						break;
					}
				}
			}
			
		}
		
		double precisionEntity = Evaluater.getPrecisionV2(countCorrectEntity, countPredictEntity);
		double recallEntity  = Evaluater.getRecallV2(countCorrectEntity, countTrueEntity);
		double f1Entity = Evaluater.getFMeasure(precisionEntity, recallEntity, 1);
		System.out.println("entity p,r,f1 are "+precisionEntity+" "+recallEntity+" "+f1Entity); 
		
		
		
	}
	
	public static void preprocess(String instance_dir, ArrayList<BiocDocument> documents, Tool tool) throws Exception {
		File fInstanceDir = new File(instance_dir);
		if(!fInstanceDir.exists()) {
			boolean ret = fInstanceDir.mkdir();
			if(ret==false)
				throw new Exception("failed to create the directory: "+fInstanceDir.getAbsolutePath());
		}
		else
			IoUtils.clearDirectory(fInstanceDir);
		
		for(int j=0;j<documents.size();j++) {
			BiocDocument document = documents.get(j);
			ArrayList<PerceptronInputData> inputDatas = new ArrayList<PerceptronInputData>();
			ArrayList<PerceptronOutputData> outputDatas = new ArrayList<PerceptronOutputData>();
			List<Sentence> mySentences = prepareNlpInfo(document, tool);
			buildInputData(inputDatas, mySentences);
			buildGoldOutputData(document, outputDatas, mySentences);
			
			File documentDir = new File(instance_dir+"/"+document.id);
			documentDir.mkdir();
			for(int k=0;k<inputDatas.size();k++) {
				ObjectSerializer.writeObjectToFile(inputDatas.get(k), documentDir+"/"+k+".input");
				ObjectSerializer.writeObjectToFile(outputDatas.get(k), documentDir+"/"+k+".output");
			}
		}
	}
	
	public static List<Sentence> prepareNlpInfo(BiocDocument document, Tool tool) {
		List<Sentence> mySentences = new ArrayList<Sentence>();
		String content = document.title+" "+document.abstractt;
		
		int offset = 0;
		// sentence segmentation
		List<String> sentences = tool.sentSplit.splitWithFilters(content);
		for(int mmm = 0;mmm<sentences.size();mmm++) {
			String sentence = sentences.get(mmm);
			
			ArrayList<Segment> given = new ArrayList<Segment>();
			ArrayList<Segment> segments = TokenizerWithSegment.tokenize(offset, sentence, given);
			List<CoreLabel> tokens = new ArrayList<CoreLabel>();
			for(Segment segment:segments) {
				if(segment.word.indexOf("'-") != -1)
					System.out.print("");
				CoreLabel token = new CoreLabel();
				token.setWord(segment.word);
				token.setValue(segment.word);
				token.setBeginPosition(segment.begin);
				token.setEndPosition(segment.end);
				tokens.add(token);
			}
			
			int sentenceLength = sentence.length();
					    
		    Sentence mySentence = new Sentence();
			mySentence.offset = offset;
			mySentence.length = sentenceLength;
			mySentence.tokens = tokens;
			
			mySentences.add(mySentence);
			
			offset += sentenceLength;
		}
		
		return mySentences;
	}		

	public static void buildInputData(ArrayList<PerceptronInputData> inputDatas,List<Sentence> mySentences) {
		// build input data
		for(Sentence mySentence:mySentences) {
			PerceptronInputData inputdata = new PerceptronInputData();
			for(int i=0;i<mySentence.tokens.size();i++) {
				CoreLabel token = mySentence.tokens.get(i);
				// build input data
				inputdata.tokens.add(token.word());
				inputdata.offset.add(token.beginPosition());
			}
			inputdata.sentInfo = mySentence;
			inputDatas.add(inputdata);
		}
	}
	
	public static void buildGoldOutputData(BiocDocument document, ArrayList<PerceptronOutputData> outDatas,List<Sentence> mySentences) {
		// build output data
		for(Sentence mySentence:mySentences) {
			PerceptronOutputData outputdata = new PerceptronOutputData(true);
			Entity entity = new Entity(null, null, 0, null, null);
			Entity oldGold = null;
			// for each token
			for(int i=0;i<mySentence.tokens.size();i++) {
				CoreLabel token = mySentence.tokens.get(i);
				// build the segments of output data begin
				Entity newGold = document.isInsideAGoldEntityAndReturnIt(token.beginPosition(), token.endPosition()-1);
				if(newGold == null) {
					if(entity.text!=null) { // save the old
						outputdata.segments.add(entity);
						entity = new Entity(null, null, 0, null, null);
					}
					// save the current, because the empty type segment has only one length.
					entity.type = Perceptron.EMPTY;
					entity.offset = token.beginPosition();
					entity.text = token.word();
					entity.start = i;
					entity.end = i;
					outputdata.segments.add(entity);
					entity = new Entity(null, null, 0, null, null);
				} else {
					if(oldGold!=newGold) { // it's a new entity
						if(entity.text!=null) { // save the old
							outputdata.segments.add(entity);
							entity = new Entity(null, null, 0, null, null);
						}
						// it's the begin of a new entity, and we set its information but don't save it,
						// because a entity may be more than one length.
						entity.type = newGold.type;
						entity.offset = token.beginPosition();
						entity.text = token.word();
						entity.start = i;
						entity.end = i;
						entity.mesh = newGold.mesh;
						
						oldGold = newGold;
					} else { // it's a old entity with more than one length
						int whitespaceToAdd = token.beginPosition()-(entity.offset+entity.text.length());
						for(int j=0;j<whitespaceToAdd;j++)
							entity.text += " ";
						// append the old entity with the current token
						entity.text += token.word();
						entity.end = i;	
					}
				}
				// build the segments of output data end
				
			}
			if(entity.text!=null) { // save the old
				outputdata.segments.add(entity);
			}
			
			outDatas.add(outputdata);
		}
	}
	
	public static boolean twoEntitiesHaveRelation(BiocDocument document, Entity entity1, Entity entity2) {
		Relation r = new Relation(null, "CID", entity1.mesh, entity2.mesh);
		if(document.relations.contains(r))
			return true;
		else 
			return false;
	
		
	}
	
}
