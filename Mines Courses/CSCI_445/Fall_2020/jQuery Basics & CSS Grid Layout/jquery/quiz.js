/*jshint esversion: 8 */
$(document).ready(function() {


  const Question = (function(category, type, difficulty, correctAnswer, incorrectAnswers, question, id) {
    this.category = category;
    this.type = type;
    this.difficulty = difficulty;
    this.correctAnswer = correctAnswer;
    this.incorrectAnswers = incorrectAnswers;
    this.question = question;
    this.answers = this.incorrectAnswers;
    this.answers.push(this.correctAnswer);
    this.ID = id;
    this.fisherYatesShuffle();
    this.HTML = '';
    this.generateHTML();
    this.answered = false;
    this.submitted = false;
    this.answerDelay = false;
    this.answeredCount = 0;
    this.generated = false;
    this.setReward();
  });

  Question.prototype.setReward = (function() {
    if(this.difficulty == "easy") this.reward = 100;
    if(this.difficulty == "medium") this.reward = 250;
    if(this.difficulty == "hard") this.reward = 500;
  });

  Question.prototype.generateHTML = (function() {

    if (this.type == "multiple" || this.type == "boolean") {
      this.HTML = `<div class="question-container ${this.type} ${this.difficulty}"><div class="question" id="${'question'+this.ID}">${this.question}</div>`;

      this.answers.forEach((answer, i) => {
        this.HTML += `<div class="answer" id="${'answer'+this.ID+i}"><input type="radio" class="radio radio-answer" id="radio-${''+this.ID+i}" name="group${this.ID}" value="${answer}"/><label for="radio-${''+this.ID+i}"></label><span class="answer-text">${answer}</span></div>`;
      });
      this.HTML += '</div>';
    } else {
      alert("Unknown Question Type");
    }
  });

  Question.prototype.fisherYatesShuffle = (function() {
    let currentIndex = this.answers.length, temporaryValue, randomIndex;
    // While there remain elements to shuffle...
    while (0 !== currentIndex) {
      // Pick a remaining element...
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex -= 1;
      // And swap it with the current element.
      temporaryValue = this.answers[currentIndex];
      this.answers[currentIndex] = this.answers[randomIndex];
      this.answers[randomIndex] = temporaryValue;
    }
  });

  const Quiz = (function() {
    this.query = "https://opentdb.com/api.php?amount=10";
    this.questions = [];
    this.HTML = '';
    this.currentQuestion = null;
    this.getHighScore();
    this.score = 0;
    this.answerCount = 0;
    this.resetTimer = true;
    this.displayCustomization();
  });

  Quiz.prototype.reset = (function() {
    this.questions = [];
    this.HTML = '';
    this.currentQuestion = null;
    this.score = 0;
    this.answerCount = 0;
    this.getHighScore();
    this.displayCustomization();
    this.query = "";
  });

  Quiz.prototype.displayCustomization = (function() {
    this.score = 0;
    $("#user-score").attr("data-score", parseInt(this.score));
    let _this = this;
    $("#quiz-customization").css({"display":"flex"});
    $("#results-page").hide();
    // On user submitting quiz form
    $("#quiz-customization-btn").on('click', function() {

      if (_this.resetTimer) {
        _this.resetTimer = false;
        setTimeout((function() {
          _this.resetTimer = true;
        }).bind(_this), 500);

        new Promise(function(resolve,reject) {
          resolve(_this);
        }).then((_this) => {
          return _this.startQuery();
        });
      }
    });
  });

  Quiz.prototype.startQuery = (async function() {
    let _this = this;
    await (new Promise(function(resolve,reject) {
      resolve(_this);
    }).then((data) => {
      return data.startQuiz();
    }).then((data) => {
      return _this;
    }));
  });


  Quiz.prototype.startQuiz = (function() {
    $("#quiz-customization").hide();
    let _this = this;
    return new Promise(function(resolve, reject) {
      resolve(_this);
    }).then((_this) => {
      return _this.formatQuery();
    }).then((_this) => {
      return _this.fetch();
    });
  });


  Quiz.prototype.gameOver = (function() {
    // Game over Animations

    $(".quiz-container").remove();
    $(".attempt").remove();
    $("#results-page").css({
      "display": "block",
      "width": "0%",
      "opacity": "0",
      "fontSize": "0rem"
    });
    $("#results-page").animate({
      width: "30%",
      opacity: 1,
    }, 750).animate({
      fontSize: "1em"
    }, 250);
    $("#quiz-retry-btn").animate({
      marginTop: "4rem",
      padding: "0.75rem"
    },1000);
    $("#result").attr("data-score", parseInt(this.score));

  });


  Quiz.prototype.addQuestion = (function(question) {
    this.questions.push(question);
  });


  Quiz.prototype.displayQuestion = (function(questionIndex){
    this.currentQuestion = questionIndex;
    $(".question").parent().hide();
    $(`#question${this.currentQuestion}`).parent().show();
  });


  Quiz.prototype.fetch = (function() {
    let _this = this;
    new Promise(function(resolve, reject) {
      resolve(_this);
    }).then((_this) => {
      return $.getJSON( _this.query, (function(data) {
        $.each( data, function( key, val ) {
          if(key == "results") {

            let quizQuestions = val;
            quizQuestions.forEach((question, i) => {
              let quizQuestion = new Question(
                question.category,
                question.type,
                question.difficulty,
                question.correct_answer,
                question.incorrect_answers,
                question.question, i
              );
            _this.addQuestion(quizQuestion);
            });
          } else if (key == "status") {
            if (val !== 200) alert("Failed to fetch quiz... :(");
          }
        });
      })).done((data) => {
        _this.HTML = '<main class="quiz-container">';
        _this.questions.forEach((question, i) => {
          _this.HTML += question.HTML;
        });
        _this.generateSumbitButton();
        _this.generateDirectory();
        _this.HTML += '</main>';
        setTimeout(function() {
          $("body").append(_this.HTML);
          _this.activateButtons();
          return _this;
        }, 10);
      });
    });
    return _this;
  });


  Quiz.prototype.generateDirectory = (function() {
    this.HTML += '<div class="quiz-directory">';
    this.questions.forEach((question, i) => {
      this.HTML += `<div class="directory-option" data-question="${question.ID}" id="directory-option${question.ID}">${question.category} for: <span class="reward">${question.reward}</span></div>`;
    });
    this.HTML += '</div>';
  });


  Quiz.prototype.generateSumbitButton = (function() {
    this.HTML += '<span id="quiz-submit-btn">Check It!</span>';
  });


  Quiz.prototype.processAnswer = (function(rightAnswer) {
    let _this = this;

    this.questions[this.currentQuestion].answers.forEach((answer, i) => {
      let answerElm = $(`#answer${this.currentQuestion}${i}`);
      if(answer == rightAnswer) {
        // mark as correct
      } else {
        // mark as incorrectAnswers
        $(answerElm).css({"opacity": 0.3});
      }
    });

    if(this.currentAnswer == rightAnswer) {
      this.score += this.questions[this.currentQuestion].reward;
      $("#user-score").attr("data-score", parseInt(this.score));
      if (this.localHighScore == null || this.score > this.localHighScore) {
        localStorage.setItem('score', this.score);
        $("#top-score").show();
        $("#top-score").attr("data-score", parseInt(this.score));
      }
      return true;
    } else {
      return false;
    }
  });


  Quiz.prototype.getHighScore = (function() {
    this.localHighScore = localStorage.getItem('score') || null;
    if (this.localHighScore !== null) {
      $("#top-score").attr("data-score", this.localHighScore);
    } else {
      $("#top-score").hide();
      localStorage.setItem("score", 0);
    }
  });


  Quiz.prototype.activateButtons = (function() {
    let _this = this;
    let attemptResult = "";
    // On user submitting answer
    $("#quiz-submit-btn").on('click', function() {
      if (!$(_this).hasClass("disabled no-click") && _this.currentQuestion != null && _this.questions[_this.currentQuestion].answered != false && !_this.questions[_this.currentQuestion].submitted) {
        let rightAnswer =       _this.questions[_this.currentQuestion].correctAnswer;
        _this.currentAnswer = $(`input:radio[name='group${_this.currentQuestion}']:checked`).val();
        if(!_this.questions[_this.currentQuestion].submitted && _this.questions[_this.currentQuestion].answered) {
          if(_this.processAnswer(rightAnswer)) {
            attemptResult = "correct";
          } else {
            attemptResult = "wrong";
          }
          let attempt = `<span class="attempt ${attemptResult}" data-question="${_this.questions[_this.currentQuestion].question} : ${_this.questions[_this.currentQuestion].correctAnswer}">${_this.questions[_this.currentQuestion].category}\t:\t  ${_this.questions[_this.currentQuestion].difficulty}</span>`;
          $("#attempts").append(attempt);
          _this.questions[_this.currentQuestion].submitted = true;
          $(`input:radio[name='group${_this.currentQuestion}']`).prop('disabled', true);
          $(".no-click").removeClass("no-click");
          $("#quiz-submit-btn").addClass("no-click");
          _this.answerCount += 1;
          console.log(_this.answerCount, _this.questions.length);
          if (_this.answerCount == _this.questions.length) _this.gameOver();
        } else {
          // Already answered question
        }
      } else {
        $("#quiz-submit-btn").addClass("invalid");
        if (!_this.answerDelay) {
          if (_this.currentQuestion != null && !_this.questions[_this.currentQuestion].answered && !_this.questions[_this.currentQuestion].submitted) {
            $("#quiz-submit-btn::after").show();

          }
          _this.answerDelay = true;
          setTimeout((function() {
            _this.answerDelay = false;
            $("#quiz-submit-btn").removeClass("invalid");
            }).bind(_this), 300);
          }
        }

    });

    // On user changing answer
    $(".radio-answer").on('change', function() {
      _this.questions[_this.currentQuestion].answered = true;
      if(!$(`#directory-option${_this.questions[_this.currentQuestion].ID}`).hasClass("disabled")) {
        $(`#directory-option${_this.questions[_this.currentQuestion].ID}`).addClass("disabled");
      }
      $("#quiz-submit-btn").removeClass("disabled no-click");
    });

    // On user changing question
    $(".directory-option").on('click', function() {
      if(!$(this).hasClass("no-click")){
        if(_this.currentQuestion == null || _this.questions[_this.currentQuestion].submitted){
          _this.displayQuestion($(this).data("question"));
          $(".directory-option").addClass("no-click");
          $(".current").removeClass("current");
          $(this).addClass("current");
          $("#quiz-submit-btn").addClass("disabled no-click");
        } else {
          // Must answer question before moving on!
        }
      }
    });

    $("#quiz-retry-btn").on('click', function() {
      _this.reset();
    });

    $("#quiz-submit-btn").addClass("disabled no-click");

    return this;
  });

  Quiz.prototype.formatQuery = (function() {
    let amount = $("#trivia_amount").val();
    this.query = "https://opentdb.com/api.php?amount="+amount;
    let category = $("#trivia-category").val();
    if (category != "any") this.query += "&category=" + category;
    let difficulty = $("#trivia-difficulty").val();
    if (difficulty != "any") this.query += "&difficulty=" + difficulty;
    let type = $("#trivia-type").val();
    if (type != "any") this.query += "&type=" + type;

    return this;
  });

  Quiz.prototype.fisherYatesShuffle = (function() {
    let currentIndex = this.questions.length, temporaryValue, randomIndex;
    // While there remain elements to shuffle...
    while (0 !== currentIndex) {
      // Pick a remaining element...
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex -= 1;
      // And swap it with the current element.
      temporaryValue = this.questions[currentIndex];
      this.questions[currentIndex] = this.questions[randomIndex];
      this.questions[randomIndex] = temporaryValue;
    }
  });

  const quiz = new Quiz();

  $(".heading, #results-text").neonText({
    textColor: "rgba(255,255,255,0)",
    textSize: '1.35rem',
    neonHighlight: 'rgba(255,255,255,0.1)',
    neonHighlightColor: 'rgb(15, 170, 235)',
    neonHighlightHover: 'rgb(15, 170, 235)',
    neonFontHover: "rgba(255,255,255,0)",
    neonSpreadFactor: 0.1
  });

  $(".footing").neonText({
    textColor: "rgba(255,255,255,0)",
    textSize: '1rem',
    neonHighlight: 'rgba(255,255,255,0.65)',
    neonHighlightColor: 'rgb(15, 170, 235)',
    neonHighlightHover: 'rgb(15, 170, 235)',
    neonFontHover: "rgba(255,255,255,0)",
    neonSpreadFactor: 0.05
  });

});
