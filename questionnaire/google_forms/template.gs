// Generated data is kept in questionnaire/private and is not committed.
const EXPERIMENTS = __EXPERIMENT_DATA__;
const RATING_PATTERN = '^[+-]?(0(?:\\.[0-9])?|1(?:\\.0)?)$';
const CRITERIA = '会話の初めから現在読んだ会話までを踏まえて、その時点での「仲の良さ（親密度）」を直感的に回答してください。\nAとB・BとC・CとAについて、−1.0〜＋1.0の実数（小数第1位まで）で評価してください。\n0.0：特に親しさも対立も感じない中立的な状態\n−1.0：強い対立（皮肉、批判、無視、相手を無視して話を進める）\n＋1.0：非常に親しい（共感、褒める、相手に話題を振る、一緒に行動する）\n半角で入力してください。例：-0.3、0、+0.2';
function createMuzintouForm() { return createExperiment_('muzintou_ari_text'); }
function createAiForm() { return createExperiment_('ai_nashi_text'); }
function transcript_(turns) { return turns.map(t => t.speaker + 'さん：' + t.text).join('\n\n'); }
function createExperiment_(id) {
  const lock=LockService.getScriptLock();lock.waitLock(30000);
  try {
    const cfg=EXPERIMENTS.find(e=>e.id===id);
    if(!cfg || !cfg.turns.length || cfg.turns.length%3)throw new Error('原稿の発話数を確認してください');
    const props=PropertiesService.getScriptProperties(),key='FORM_'+id;
    let saved=JSON.parse(props.getProperty(key)||'null');
    if(!saved){
      const form=FormApp.create('【文字版】'+cfg.theme+' 関係性アンケート',false);
      saved={form:form.getId(),next:0,count:0,ready:false};props.setProperty(key,JSON.stringify(saved));
    }
    const form=FormApp.openById(saved.form);
    if(saved.ready){logLinks_(form,saved);return;}
    // Only unfinished, unpublished draft items after the last checkpoint are rebuilt.
    if(form.isPublished() || form.getResponses().length)throw new Error('作成途中のフォームが公開・回答済みです。手動で確認してください');
    while(form.getItems().length>saved.count)form.deleteItem(form.getItems().length-1);
    if(saved.count===0){
      form.setDescription(cfg.description+'\n\n'+CRITERIA+'\n\n3発話ごとにページが進みます。「戻る」で前のページを確認・修正できます。最後に「送信」を押して回答を確定してください。')
        .setProgressBar(true).setShuffleQuestions(false).setCollectEmail(false)
        .setAllowResponseEdits(true).setPublishingSummary(false)
        .setConfirmationMessage('回答を保存しました。ご協力ありがとうございました。');
      form.addTextItem().setTitle('回答者ID').setHelpText('担当者から指定された数字のIDを入力してください。').setRequired(true)
        .setValidation(FormApp.createTextValidation().requireTextMatchesPattern('^[0-9]{1,30}$').setHelpText('半角数字で入力してください。').build());
      form.addMultipleChoiceItem().setTitle('テーマ').setChoiceValues([cfg.theme]).setRequired(true);
      saved.count=form.getItems().length;props.setProperty(key,JSON.stringify(saved));
    }
    const validation=FormApp.createTextValidation().requireTextMatchesPattern(RATING_PATTERN).setHelpText('半角で-1.0〜+1.0、小数第1位まで入力してください。例：-0.3、0、+0.2').build();
    for(let i=saved.next;i<cfg.turns.length/3;i++){
      form.addPageBreakItem().setTitle((i+1)+'. 発話 '+(i*3+1)+'〜'+(i*3+3));
      if(i>0)form.addSectionHeaderItem().setTitle('前の3発話').setHelpText(transcript_(cfg.turns.slice((i-1)*3,i*3)));
      form.addSectionHeaderItem().setTitle('今回の3発話').setHelpText(transcript_(cfg.turns.slice(i*3,i*3+3)));
      for(const pair of ['AとB','BとC','CとA'])form.addTextItem().setTitle((i+1)+'. '+pair).setHelpText('−1.0〜＋1.0（小数第1位まで）。0.0は中立。').setRequired(true).setValidation(validation);
      saved.next=i+1;saved.count=form.getItems().length;props.setProperty(key,JSON.stringify(saved));
    }
    form.addParagraphTextItem().setTitle('感想（任意）');
    let sheet;
    if(saved.sheet)sheet=SpreadsheetApp.openById(saved.sheet);
    else{sheet=SpreadsheetApp.create('【文字版回答】'+cfg.theme);saved.sheet=sheet.getId();props.setProperty(key,JSON.stringify(saved));}
    form.setDestination(FormApp.DestinationType.SPREADSHEET,sheet.getId());
    saved.ready=true;props.setProperty(key,JSON.stringify(saved));logLinks_(form,saved);
  } finally {lock.releaseLock();}
}
function logLinks_(form,saved){
  console.log('編集URL：'+form.getEditUrl());
  console.log('回答URL（編集画面で公開してから配布）：'+form.getPublishedUrl());
  if(saved.sheet)console.log('回答スプレッドシート：https://docs.google.com/spreadsheets/d/'+saved.sheet+'/edit');
  console.log('フォームは未公開で作成します。編集画面で確認後、公開してください。既存フォームは再実行しても複製しません。');
}
