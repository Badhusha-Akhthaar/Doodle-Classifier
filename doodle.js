const len=784;
const total_data=1000;
let cats_data;
let camels_data;
let candles_data;
let clouds_data;
let fences_data;
let guitars_data;
let headphones_data;
const CAT=0;
const CAMEL=1;
const CANDLE=2;
const FENCE=3;
const GUITAR=4;
const HEADPHONE=5;
let nn;
let indexNames=["I think is a Cat","I think is a Camel","I think is a Candle","I think is a Fence","I think is a Guitar","I think is a HeadPhone"];
let cats={};
let camels={};
let candles={};
let clouds={};
let fences={};
let guitars={};
let headphones={};
let isLoading=true;
function preload()
{
	camels_data=loadBytes('data_bin/camel.bin');
	cats_data=loadBytes('data_bin/cat.bin');
	candles_data=loadBytes('data_bin/candle.bin');
	fences_data=loadBytes('data_bin/fence.bin');
	guitars_data=loadBytes('data_bin/guitar.bin');
	headphones_data=loadBytes('data_bin/headphones.bin');
}
function trainEpoch(training)
{
	shuffle(training,true);
	for(let i=0;i<training.length;i++)
	{
		
		let data=training[i];
		let inputs=Array.from(data).map(x=>x/255);
		let label=training[i].label;
		let targets=[0,0,0,0,0,0];
		targets[label]=1;
		nn.train(inputs,targets);
	}
}
function setup()
{
	var cnv=createCanvas(280,280);
	var x = (windowWidth - width) / 2;
  	var y = (windowHeight - height) / 2;
  	cnv.position(x, y);
  	background(255);


	prepareData(camels,camels_data,CAMEL);
	prepareData(candles,candles_data,CANDLE);
	prepareData(cats,cats_data,CAT);
	prepareData(fences,fences_data,FENCE);
	prepareData(guitars,guitars_data,GUITAR);
	prepareData(headphones,headphones_data,HEADPHONE);
	nn=new NeuralNetwork(784,64,6);
	
	let training=[];
	training=training.concat(camels.training);
	training=training.concat(candles.training);
	training=training.concat(cats.training);
	training=training.concat(fences.training);
	training=training.concat(guitars.training);
	training=training.concat(headphones.training);
	let testing=[];
	testing=testing.concat(camels.testing);
	testing=testing.concat(candles.testing);
	testing=testing.concat(cats.testing);
	testing=testing.concat(fences.testing);
	testing=testing.concat(guitars.testing);
	testing=testing.concat(headphones.testing);
	for(let i=1;i<=5;i++)
	{
		trainEpoch(training);
		console.log("Epoch "+i);
	}
	isLoading=false;
	console.log("Finished");
	let percent=testAll(testing);
	console.log(percent);
	let testG=select("#guess");
	let clearB=select("#clear");
	clearB.position(x+135,y+300);
	clearB.mousePressed(function(){
		background(255);
	});
	
}
let flag=0;
function draw()
{
	
	strokeWeight(8);
	stroke(0);
	if(mouseIsPressed)
	{
		line(pmouseX,pmouseY,mouseX,mouseY);
		flag=1;
	}
	if(!mouseIsPressed && flag==1)
	{
		let inputs=[];
		let img=get();
		img.resize(28,28);
		img.loadPixels();
		for(let i=0;i<len;i++)
		{
			let bright=img.pixels[i * 4];
			inputs[i]=(255 - bright) / 255.0;

		}
		let guess=nn.feedforward(inputs);
		let m=max(guess);
		let classification=guess.indexOf(m);
		let post=indexNames[classification];
		var x = (windowWidth - width) / 2;
  		var y = (windowHeight - height) / 2;
  		var disp=document.getElementById("disp");
  		disp.innerHTML="<h4 ><b>"+post+"</b></h4>";
  		let sr=select("#disp");
  		sr.position(x+100,y+350);


	}	
}
function testAll(testing)
{
	let correct=0;
	for(let i=0;i<testing.length;i++)
	{
		
		let data=testing[i];
		let inputs=Array.from(data).map(x=>x/255);
		let label=testing[i].label;
		let guess=nn.feedforward(inputs);
		let m=max(guess);
		let classification=guess.indexOf(m);
		if(classification===label)
		{
			correct++;
		}

	}
	let percent=correct/testing.length;
	return percent;
}
function prepareData(category,data,label)
{
	category.training=[];
	category.testing=[];
	for(let i=0;i < total_data; i++)
	{
		let offset=i*len;
		let threshold=floor(0.8*total_data);
		if(i<threshold)
		{
			category.training[i]=data.bytes.subarray(offset,offset+len);
			category.training[i].label=label;
		}
		else
		{
			category.testing[i-threshold]=data.bytes.subarray(offset,offset+len);
			category.testing[i-threshold].label=label;

		}
	}
}