// Google Apps Script: paste this entire file into a NEW project and run createVideoForm.
// Copies the exact source form referenced in the supplied PDF. Never edits the source.
const VIDEO_SOURCE_FORM_ID = '1Jp1GUBGhMsmAjBsbC9IUBgp7DuukinhBjxHgBkgbygo';

function videoWording_(text) {
  return (text || '').replace(/【文字版】/g, '【動画版】')
    .replace(/現在読んだ会話まで/g, '現在視聴した会話まで')
    .replace(/3発話ごとにページが進みます。/g,
      '動画が停止したら、表示されているインデックスと同じ番号のページで回答してください。回答後はフォームの次のページへ進み、動画の「次の区間を再生」を押してください。最後のインデックス41の回答後は、自由記述に進んでください。');
}

function videoQuestions_(form) {
  return form.getItems().filter(item => ![
    FormApp.ItemType.SECTION_HEADER, FormApp.ItemType.PAGE_BREAK
  ].includes(item.getType())).map(item => ({
    title: item.getTitle(), type: String(item.getType()), help: item.getHelpText()
  }));
}

function createVideoForm() {
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    const props = PropertiesService.getScriptProperties();
    const key = 'VIDEO_FORM_' + VIDEO_SOURCE_FORM_ID;
    let saved = JSON.parse(props.getProperty(key) || 'null');
    const source = FormApp.openById(VIDEO_SOURCE_FORM_ID);
    const questions = videoQuestions_(source);
    const ratings = questions.filter(q => /^\d+\. (AとB|BとC|CとA)$/.test(q.title));
    if (ratings.length !== 123) throw new Error('元フォームの評定欄が123問ではありません。元フォームを確認してください。');
    if (!saved) {
      const copy = DriveApp.getFileById(VIDEO_SOURCE_FORM_ID).makeCopy(videoWording_(source.getTitle()));
      saved = {id: copy.getId(), ready: false};
      props.setProperty(key, JSON.stringify(saved));
    }
    if (saved.id === VIDEO_SOURCE_FORM_ID) throw new Error('元フォームは変更できません。');
    const form = FormApp.openById(saved.id);
    if (!saved.ready) {
      if (form.getResponses().length) throw new Error('コピーに回答があります。変更せず終了します。');
      // Finish as an unpublished draft. Only the copied form is changed.
      if (form.supportsAdvancedResponderPermissions()) form.setPublished(false);
      else form.setAcceptingResponses(false);
      form.setTitle(videoWording_(source.getTitle()));
      form.setDescription(videoWording_(source.getDescription()));
      const items = form.getItems();
      for (let i = items.length - 1; i >= 0; i--) {
        const item = items[i];
        if (item.getType() === FormApp.ItemType.SECTION_HEADER) {
          const section = item.asSectionHeaderItem();
          if (/^(前の3発話|今回の3発話)$/.test(section.getTitle().trim())) {
            form.deleteItem(i);
          } else {
            section.setHelpText(videoWording_(section.getHelpText()));
          }
        } else if (item.getType() === FormApp.ItemType.PAGE_BREAK) {
          const page = item.asPageBreakItem();
          page.setHelpText(videoWording_(page.getHelpText()));
        }
      }
      if (JSON.stringify(videoQuestions_(form)) !== JSON.stringify(questions)) {
        throw new Error('質問項目が一致しません。未公開のコピーを確認してください。');
      }
      const pages = form.getItems(FormApp.ItemType.PAGE_BREAK);
      if (pages.length !== 41) throw new Error('区間ページ数が41ではありません。未公開のコピーを確認してください。');
      // Give the copied questionnaire its own response destination.
      if (!saved.sheet) {
        saved.sheet = SpreadsheetApp.create('【動画版回答】無人島ー縛りあり').getId();
        props.setProperty(key, JSON.stringify(saved));
      }
      form.setDestination(FormApp.DestinationType.SPREADSHEET, saved.sheet);
      saved.ready = true;
      props.setProperty(key, JSON.stringify(saved));
    }
    console.log('動画版の編集URL：' + form.getEditUrl());
    console.log('動画版の回答URL（公開後に配布）：' + form.getPublishedUrl());
    console.log('動画版の回答保存先：' + SpreadsheetApp.openById(saved.sheet).getUrl());
    console.log('未公開で作成しました。プレビューで確認後、編集画面から公開してください。');
  } finally {
    lock.releaseLock();
  }
}
