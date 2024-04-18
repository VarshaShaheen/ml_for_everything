import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from flask import Flask, request, render_template, jsonify
from text_summarisation import summarize

if __name__ == '__main__':
    reference = "The content is a summary of various articles and reports. It includes a highly classified assessment on an influence campaign, an analysis of Russian hacking, a story about a parent's dedication to their child's education, a discussion on the Affordable Care Act, an evaluation of the Obama administration's foreign policy, a review of"
    input_text = "washington congressional republican new fear come health care lawsuit obama administration might win incoming trump administration could choose longer defend executive branch suit challenge administration authority spend billion dollar health insurance subsidy american handing house republican big victory issue sudden loss disputed subsidy could conceivably cause health care program implode leaving million people without access health insurance republican prepared replacement could lead chaos insurance market spur political backlash republican gain full control government stave outcome republican could find awkward position appropriating huge sum temporarily prop obama health care law angering conservative voter demanding end law year another twist donald trump administration worried preserving executive branch prerogative could choose fight republican ally house central question dispute eager avoid ugly political pileup republican capitol hill trump transition team gaming handle lawsuit election put limbo least late february united state court appeal district columbia circuit yet ready divulge strategy given pending litigation involves obama administration congress would inappropriate "
    summary = summarize(input_text)
    candidate = summary  # Model's output
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    # Calculate BLEU score without smoothing
    score = sentence_bleu([reference_tokens], candidate_tokens)
    print("BLEU score:", score)
