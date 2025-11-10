Usage
=====

For the evaluation, you can simply run::

  mcif_eval -t {short/long} -l {en/de/it/zh} \
      -s model_outputs.xml -r MCIF1.0.IF.{short/long}.{en/de/it/zh}.ref.xml

where `model_outputs.xml` contains the outputs of your model for the selected track or context
length (`short` or `long`) and target language among English (`en`), German (`de`), Italian (`it`)
and Chinese (`zh`), and is structured as follows::

  <?xml version='1.0' encoding='utf-8'?>
  <testset name="MCIF" type="output">
    <task track="{short/long}" text_lang="{en/de/it/zh}">
      <sample id="1">{SAMPLE1_CONTENT}</sample>
      <sample id="2">{SAMPLE2_CONTENT}</sample>
     ....
    </task>
  </testset>

To ease usability, we provide a helper function
(`mcif.io.write_output <https://github.com/hlt-mt/mcif/blob/main/src/mcif/io.py>`__) that
automatically formats model predictions into the XML structure required by the MCIF evaluation
script.
The method takes as input:

- ``samples``: a list of ``mcif.io.OutputSample`` containing the sample id and its related prediction;
- ``track``: the context length or track (``short/long``);
- ``language``: the target language (``en/de/it/zh``);
- ``output_name``: the semantic name of the output (e.g. ``My model``);
- ``output``: a path or a byte buffer where the XML file containing all system's outputs, ready for
  evaluation, is written.
