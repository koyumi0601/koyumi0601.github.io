var g_oConvert_gnb = "fw";                          // 정방향, 역방향 값
var isArk_gnb = true;                               // 자동완성 기능 사용 여부
var isKeydown_gnb = false;                          // 브라우저가 파이어폭스, 오페라일 경우 keydown 사용 여부
var isListShow_gnb = true;
var cursorPos_gnb = -1;                             // 자동완성 커서 위치 값
var formName_gnb = "#search_gnb";                       // 검색 form의 name을 설정한다.
var queryId_gnb = "#topSearch";                         // 검색어 <input> 의 id을 설정한다
var arkId_gnb = "#ark_gnb";                             // 자동완성 전체 <div> 의 id을 설정한다
var wrapId_gnb = "preview_wrap";                        // 자동완성 결과 <div> 의 id을 설정한다
var totalFwCount_gnb = 0;                           // 전방 검색 전체 개수
var totalRwCount_gnb = 0;                           // 후방 검색 전체 개수
var totalArkCount_gnb = 0;                          // 신규
var target_gnb = "dx";                              // ARK 웹서버 설정파일의 목록에 있는 추천어 서비스 대상을 지정한다.
var charset_gnb = "utf-8";                          // 인코딩 설정 (인코딩이 utf-8이 아닐 경우 8859_1 로 설정해야함)
var datatype_gnb = "json";                          // 반환받을 Data의 타입을 설정. XML 과 JSON이 가능 (xml | json)
var arkPath = "/solution/searchApi/ark";                     // 자동완성 경로
var transURL = arkPath + "/ark_trans.jsp";      // trans 페이지의 URL을 설정한다.
var tempQuery_gnb = "";
var keyword = "";
var _arr = [];
var _re_search_json     = "/solution/searchApi/search_json.jsp";
//var _re_search_json     = "/solution/searchApi/search_json2.jsp"; // 개발용
var tempLocation = location.href;
/**
 *  ARK 구성요소의 위치 및 크기를 아래 변수를 통해 조정함.
 */

var browser = "";
if ($.browser.msie) {
    browser = "IE";
} else if ($.browser.mozilla) {
    browser = "FF";
} else if ($.browser.opera) {
    browser = "OPERA";
} else if ($.browser.webkit) {
    browser = "CHROME";
}

var browserVersion = $.browser.version;         // 웹브라우져의 버전
var offset = null;
var offsetTop = 0;
var offsetLeft = 0;

/** ARK 구성요소의 위치 및 크기를 아래 변수를 통해 조정함. **/
var IE6_gnb_OFFSET = -36;               // IE6 일 경우 TOP 옵셋 값 오차 조정
var IE6_LEFT_OFFSET = 20;               // IE6 일 경우 LEFT 옵셋 값 오차 조정
var IE7_gnb_OFFSET = -63;               // IE7 일 경우 TOP 옵셋 값 오차 조정
var IE7_LEFT_OFFSET = -36;              // IE7 일 경우 LEFT 옵셋 값 오차 조정
var IE8_gnb_OFFSET = 0;                 // IE8 일 경우 TOP 옵셋 값 오차 조정
var IE8_LEFT_OFFSET = 0;                // IE8 일 경우 LEFT 옵셋 값 오차 조정
var FF_gnb_OFFSET = 0;
var FF_LEFT_OFFSET = 0;
var CHROME_gnb_OFFSET = 0;
var CHROME_LEFT_OFFSET = 0;
var OPERA_gnb_OFFSET = 0;
var OPERA_LEFT_OFFSET = 0;

/** ARK 구성요소의 위치 및 크기를 아래 변수를 통해 조정함. **/
var arkWidth_gnb = 37;                              // 자동완성 전체 넓이 값을 설정한다(변동폭).
var arkTop_gnb = 29;                                // 자동완성 상단에서의 위치 값을 설정한다.
var arkLeft_gnb = -3;                               // 자동완성 왼쪽에서의 위치 값을 설정한다.
var arkImgTop_gnb = 0;                              // 자동완성 화살표 이미지의 상단에서 위치 값을 설정한다.
var arkImgLeft_gnb = 0;                         // 자동완성 화살표 이미지의 왼쪽에서 위치 값을 설정한다.
var tooltip01TopPos_gnb = 0;                        // 자동완성 기능끄기 툴팁의 상단 기준 위치 오차 조정값
var tooltip01LeftPos_gnb = -155;                    // 자동완성 기능끄기 툴팁의 좌측 기준 위치 오차 조정값
var tooltip02TopPos_gnb = 0;                        // 자동완성 기능켜기 툴팁의 상단 기준 위치 오차 조정값
var tooltip02LeftPos_gnb = 0;                       // 자동완성 기능켜기 툴팁의 좌측 기준 위치 오차 조정값

$(document).ready(function() {
    // 자동완성 기능 사용 여부 확인 한다.
    $(queryId_gnb).attr("autocomplete","off");

    $(queryId_gnb).mouseout(function (){
        //$(this).blur();
     });


    $(queryId_gnb).focus(function (){
         if ($(queryId_gnb).val() =="" ){

             viewWordRecent();
             $(".recent_popular_keyword").show();
         } else {

             requestCollection($(queryId_gnb).val());
         }
		 $(queryId_gnb).val(''); // 2017-04-27 포커싱때 검색어 clear
     });
    // 브라우져별 ARK 옵셋 설정
    setArkOffset_gnb();

    if($.browser.opera || $.browser.mozilla) {
        $(document).keydown(function(event) {
            if ($(event.target).is(queryId_gnb)) {
                if (event.which == 38 || event.which == 40) {
                    if (query != "") {
//                        showArk_gnb();
                    }
                    moveFocusEvent_gnb(event);
                } else {
                    if ($(event.target).is(queryId_gnb)) {
                        isKeydown_gnb = true;
                        eventKeydown_gnb();
                    }
                }

                var query = $(queryId_gnb).val();
                //포커스 갔을 시에 focused, on 클래스 추가
                if (query == "") {
                    $('.box_search_input').removeClass('focused on');
                    $(".keyword_list").empty();
                    
                    $(".recent_searches").show();
                    $(".recent_popular_keyword").show();
                }
                else{
                    $('.box_search_input').addClass('focused on');
                }				
            }
        });
    }else if($.browser.msie || $.browser.webkit || $.browser.chrome ||$.browser.name =="unknown") {
        $(document).keyup(function(event) {
            if($(event.target).is(queryId_gnb)) {
                
                if(event.keyCode == 38 || event.keyCode == 40) {
                    // 아래(40), 위(38) 방향키 조작시의 이벤트 처리
                    if(query != "") {
//                        showArk_gnb();
                    }
                    moveFocusEvent_gnb(event);
                }else if(event.keyCode == 16) {
                }else if(event.keyCode == 27) {
                    // ESC 키 입력시
                    if (tempLocation.indexOf("CPCSEARCH") > -1) {
                        $(".gnb_dimmed").click();
                        $(queryId_gnb).blur();
                    }else {
                        $( '.btn_search_open' ).click();
                    }
                }else if(event.keyCode == 8 && query == "") {


                    viewWordRecent();

                    $(".recent_popular_keyword").show();
                    $('#btnAuto').removeClass('on');
                }else{
                    if (isArk_gnb && $(queryId_gnb).val() != "") {
                        if(datatype_gnb == "json") {
                            requestCollection($(queryId_gnb).val());
                        }
                    }else if($(queryId_gnb).val() == "") {

                        viewWordRecent();

                        $(".recent_popular_keyword").show();
                        $('#btnAuto').removeClass('on');
                    }
                }
				
                var query = $(queryId_gnb).val();
                //포커스 갔을 시에 focused, on 클래스 추가
                if (query == "") {
                    $('.box_search_input').removeClass('focused on');
                    $(".keyword_list").hide();
                    
                    $(".recent_searches").show();
                    $(".recent_popular_keyword").show();
                }
                else{
                    $('.box_search_input').addClass('focused on');
                }
                				
            }
        });
    }

        // 브라우저에서 일어나는 클릭 이벤트를 체크한다.
    $(document).click(function(event) {
        stopEventBubble(event);
        if($(event.target).is(queryId_gnb)) {
            var query = $(queryId_gnb).val();
            viewGnb();
            if (query != "") {
                if(datatype_gnb == "json") {
                    requestCollection($(queryId_gnb).val());
                }
                keyword = query;
                isKeydown_gnb = true;
            }else {
                viewWordRecent();

                $(".recent_popular_keyword").show();
            }
          } else if ($(".header_search").find(event.target).length == 0){
              if($(event.target).attr("id")=="srh"){
                  return;
              }
              hideGnb();
          }
    });


});
function fnAutoReplace(txt, category) {
    var tmp      = '';
    var startTag = '';
    var endTag   = '';

    switch(category) {
        case "re_event" :
           startTag = '<strong>';
           endTag   = '</strong>';
            break;
        case "remove" :
            startTag = '';
            endTag   = '';
             break;
        default :
            startTag = '<strong>';
            endTag   = '</strong>';
           break;
    }

    tmp = txt.replace(/<!HS>/gi, startTag);
    tmp = tmp.replace(/<!HE>/gi, endTag);
    tmp = tmp.replace(/&gt;/gi, ' &gt; ');

    return tmp;
}


/************************************************
 * jQuery Event Bubbling 방지를 위한 함수.
 * @name stopEventBubble
 * @param evt 페이지 이벤트
 ************************************************/
function stopEventBubble(evt) {
    var eventReference = (typeof evt !== "undefined") ? evt : event;
    if(eventReference.stopPropagation) {
        eventReference.stopPropagation();
    }else{
        eventReference.cancelBubble = true;
    }
}


/************************************************
 * Browser별로 기준점 Offset 값을 설정
 * @name setArkOffset
 ************************************************/

function setArkOffset_gnb() {
    offset = $(queryId_gnb).position();

    if(offset) {
        offsetTop = offset.top;
        offsetLeft = offset.left;
    }
    else{
        offsetTop = 0;
        offsetLeft = 0;
    }

    if(browser == "IE") {
        if(browserVersion == "6.0") {
            offsetTop = offsetTop + IE6_gnb_OFFSET;
            offsetLeft = offsetLeft + IE6_LEFT_OFFSET;
        }else if(browserVersion == "7.0") {
            offsetTop = offsetTop + IE7_gnb_OFFSET;
            offsetLeft = offsetLeft + IE7_LEFT_OFFSET;
            offsetTop = offsetTop - 2;
        }else if(browserVersion == "8.0") {
            offsetTop = offsetTop + IE8_gnb_OFFSET;
            offsetLeft = offsetLeft + IE8_LEFT_OFFSET;
        }
    }else if(browser == "FF") {
        offsetTop = offsetTop + FF_gnb_OFFSET;
        offsetLeft = offsetLeft + FF_LEFT_OFFSET;
    }else if(browser == "CHROME") {
        offsetTop = offsetTop + CHROME_gnb_OFFSET;
        offsetLeft = offsetLeft + CHROME_LEFT_OFFSET;
    } else if(browser == "OPERA") {
        offsetTop = offsetTop + OPERA_gnb_OFFSET;
        offsetLeft = offsetLeft + OPERA_LEFT_OFFSET;
    }
}


function hc_replaceAll_gnb( str, s_org, s_rep )
{
    return str.split(s_org).join(s_rep);
}
function fnApplyCard(reg_url){
    if (reg_url.indexOf("cardWcd=KDBX") >-1) { //KDB대우증권 현대카드X는 대우 증권 홈페이지에서 발급
        window.open('http://www.kdbdw.com/mframe.jsp?url=/hku/hku4189/n01.do');
    } else {               //KDB대우증권 현대카드X 외 나머지 카드는 현대카드 홈페이지에서 발급
        document.location.href = reg_url;
    }
    return false;
}

function requestCollection(query){
    $("#keyword_list > li").remove();

    //$.getJSON(_re_search_json, {query:query, collection:"re_menu,re_event,re_ark", listCount:"10"}, function(data){
    $.getJSON(_re_search_json, {query:query, collection:"re_ark"}, function(data){
        var keyWordCnt = 0;
        var ark_Key = "";
        var event_key = "";
        var gb = ""; //CARD, EVENT, CULTURE
        var cnt = 0;
        var kCnt = 0;

        //var omniCount = 0;

        $.each(data.SearchQueryResult.Collection, function(idx, row){
            var keywordStr ="" ;
            var prodStr ="" ;
            var menuStr = "";
            var eventStr = "";

            if (row.Id == 're_ark'){
                var array = row.DocumentSet.Document;

                if(array != null && array.length > 0 ){
                    ark_Key = array[0].Field.DOCID;
                    $("#keyword_list > li").remove();
                    keyWordCnt = 7;

                    for(var i =0 ; i < array.length && i <keyWordCnt ; i++){
                        keywordStr += "<li id='autoKeyArr_"+i+"'><a href='javascript:keyWordClick(\""+i+"\")';>"+hc_replaceAll_gnb(hc_replaceAll_gnb(fnAutoReplace(array[i].Field.NAME,'re_ark'),'<BR>',' '),'<br>',' ')+"</a></li>";
                        cnt++;
                        kCnt++;
                    }
                    $(keywordStr).appendTo("#keyword_list");
                }
            }
        });

        if (kCnt == 0) {
            $("#keyword_list > li").remove();
            $("#keyword_list").hide();
            
            var query = $(queryId_gnb).val();
            if (query == "") {
                $(".recent_searches").show();
                $(".recent_popular_keyword").show();
            }else{
                $(".recent_searches").hide();
                $(".recent_popular_keyword").hide();
            }
        }else {
            $("#keyword_list").show();
            $(".recent_searches").hide();
            $(".recent_popular_keyword").hide();
        }
        
    }).fail(function(data){
        // 통신실패시
        $("#preview_wrap").hide();

        $('#btnAuto').removeClass('on');
    });
}
function keyWordClick(index){
    selectSearchWord($("#autoKeyArr_"+index).text());
}
/************************************************
 * 자동완성 결과 요청
 * @name requestArk
 * @param query 키보드 입력된 문자열
 ************************************************/
function requestArkJson_gnb(query) {
    jQuery.support.cors = true;
    cursorPos_gnb = -1;
    $.ajaxSetup({cache:false});
    $.ajax({
        url: transURL,
        type: "POST",
        dataType: "json",
        data: {"convert":g_oConvert_gnb, "target":target_gnb, "charset":charset_gnb, "query":hc_replaceAll_gnb(query, ' ', '`'), "datatype": datatype_gnb},
        success: function(data) {
            if(data.result.length <= 0) {
                totalFwCount_gnb = 0;
                totalRwCount_gnb = 0;
            }

            var str = "";      //asis에 있는 이것 str += "<ul class=\"txt-list txt-list-link list\">" 이 빠져있음;
            var resultArr = new Array();      //신규

            $.each(data.result, function(i, result) {
                var totalCount = parseInt(result.totalcount);
                if (i == 0) {
                    totalFwCount_gnb = totalCount;
                } else {
                    totalRwCount_gnb = totalCount;
                }

                if (totalCount > 0) {
                    // 자동완성 리스트 설정
                    $.each(result.items, function(num,item){     // 많이 달라졌음.
                        if (i != 0) {
                            num = totalFwCount_gnb + num;
                        }
                        // 반드시 두번째 항목에 자동완성 키워드가 와야함.
                        var result = { id : num,
                                        keyword : item.keyword,
                                        hkeyword : item.hkeyword,
                                        linkname : item.linkname,
                                        linkurl : item.linkurl,
                                        weight : item.count,
                                        type : item.type };
                        resultArr.push(result);
                    });
                }
            });

            totalArkCount_gnb = resultArr.length;

            if (totalArkCount_gnb == 0) {//기 asis = if ((totalFwCount_gnb + totalRwCount_gnb) == 0) {
                hideArk_gnb();
            }else if( totalArkCount_gnb > 0 ){
                /* Keyword List*/
                var keywordStr ="" ;
                var prodStr ="" ;
                var gb = ""; //CARD, EVENT, CULTURE
                $("#keyword_list > li").remove();
                for(var i =0 ; i < totalArkCount_gnb && i <5 ; i++){
                    keywordStr += "<li><a href='"+resultArr[i].linkurl+"'>"+resultArr[i].keyword+"</a></li>";
                }
                $(keywordStr).appendTo("#keyword_list");
                

                showArk_gnb();
            }
        }
    });
}




/************************************************
 * 브라우저가 FireFox, Opera 일 경우 한글 입력
 * @name eventKeydown
 ************************************************/
function eventKeydown_gnb() {
    // 방향키 이동시 메소드 실행을 중지시킨다.
    if(!isKeydown_gnb) {
        return;
    }

    if (keyword != $(queryId_gnb).val()) {
        keyword = $(queryId_gnb).val();
        if (keyword != "" && isArk_gnb) {
            if(datatype_gnb == "json") {
                //requestArkJson_gnb($(queryId_gnb).val());
                requestCollection($(queryId_gnb).val());
            }
        } else {
            hideArk_gnb();
        }
    }
    setTimeout("eventKeydown_gnb()", 20);
}

/************************************************
 * 방향키 이벤트 처리
 * @name moveFocusEvent_gnb
 * @param event 페이지 이벤트
 ************************************************/

function moveFocusEvent_gnb(event) {
    isKeydown_gnb = false;

    if (event.keyCode == 38) {
        if (cursorPos_gnb==-1 || cursorPos_gnb==0) {
            cursorPos_gnb = -1;
            hideArk_gnb();
            $(queryId_gnb).val(tempQuery_gnb);
            tempQuery_gnb = "";
        } else {
            onMouseOutKeyword_gnb(cursorPos_gnb);
            var currentPos_gnb = cursorPos_gnb;
            cursorPos_gnb = cursorPos_gnb - 1;
            onMouseOverKeyword_gnb(cursorPos_gnb);
//            $(queryId_gnb).val($("#gnb_f" + cursorPos_gnb).text());
            // 커서가 움직일 때 해당 항목에 음영 출력
//            $("#gnb_bg" + currentPos_gnb + " a").attr("class", "");
//            $("#gnb_bg" + cursorPos_gnb + " a").attr("class", "on");
        }
//        detailView_gnb(cursorPos_gnb);
    } else if (event.keyCode == 40) {
        if(cursorPos_gnb == -1) {
            tempQuery_gnb = $(queryId_gnb).val();
        }
        if (totalArkCount_gnb > (cursorPos_gnb + 1)) {
            onMouseOutKeyword_gnb(cursorPos_gnb);
            var currentPos_gnb = cursorPos_gnb;
            cursorPos_gnb = cursorPos_gnb + 1;
            onMouseOverKeyword_gnb(cursorPos_gnb);
//            $(queryId_gnb).val($("#gnb_f" + cursorPos_gnb).text());
            // 커서가 움직일 때 해당 항목에 음영 출력
//            $("#gnb_bg" + currentPos_gnb + " a").attr("class", "");
//            $("#gnb_bg" + cursorPos_gnb + " a").attr("class", "on");
        }
//        detailView_gnb(cursorPos_gnb);
    }
}

/************************************************
 * MouseOver 일 경우 선택한 배경을 설정
 * @name onMouseOverKeyword_gnb
 * @param cursorNum 커서의 위치 인덱스 값
 ************************************************/

function onMouseOverKeyword_gnb(cursorNum) {
    clearCursorPos_gnb();
    cursorPos_gnb = cursorNum;
//    $("#bg" + cursorNum).css({"backgroundColor" : "#eeeeee"});
//    $("#bg" + cursorNum).css({"cursor" : "pointer"});
}
/************************************************
 * MouseOut 일 경우 설정한 배경을 초기화
 * @name onMouseOutKeyword_gnb
 * @param cursorNum 커서의 위치 인덱스 값
 ************************************************/
function onMouseOutKeyword_gnb(curSorNum) {
    cursorPos_gnb = curSorNum;
//    $("#bg" + cursorPos_gnb).css({"backgroundColor" : "#ffffff"});
}
/************************************************
 * 커서 위치가 변경될 때마다 선택되지 않은 부분 초기화
 * @name clearCursorPos_gnb
 ************************************************/
function clearCursorPos_gnb() {
    for(var i=0; i<totalArkCount_gnb; i++){                                    //변경됨
//        $("#bg" + i).css({"backgroundColor" : "#ffffff"});
    }
}
/************************************************
 * 마우스 클릭시 검색을 수행
 * @name onClickKeyword_gnb
 * @param cursorPos_gnb 커서의 위치
 ************************************************/
function onClickKeyword_gnb(cursorPos_gnb) {
    $(queryId_gnb).val($("#gnb_f" + cursorPos_gnb).text());
    $(formName_gnb).submit();
}
function viewGnb(){
    var $this = $(this);
    _idx = $this.parent().index();
    $('#header').addClass('dimm');
    $( '#localMenu' ).find('li').removeClass('on');
    _bgHgt = _arr[_idx];
    //검색버튼 클릭
    if (!$('.dimmed').length) {
        $('#header').before( '<span class="dimmed gnb_dimmed"></span>');
        setTimeout(function(){
            $('.dimmed').css({ opacity : '1' });
        },0);
    }

    $this = $('.btn_search_open');
    $this.addClass('on');

    $('.bg_gnb').stop().css({ height : _bgHgt });
    $this.parent().addClass('on');
}

function hideGnb(){

    if (tempLocation.indexOf("CPCSEARCH") > -1) {
        $this = $('.btn_search_open');
        $this.removeClass('on');
        if ($(queryId_gnb).val() !=""){

            $(".recent_popular_keyword").hide();
            $("#search_word_recent").hide();
            $('#btnAuto').removeClass('on');
        }else {

            viewWordRecent();
            $(".recent_popular_keyword").show();
        }
    } else {
        $this = $('.btn_search_open');
        $this.removeClass('on');
        $('.section_search').hide();
        $('#btnAuto').removeClass('on');
    }

}

/************************************************
 * 자동완성 목록을 화면에 보여줌
 * @name showArk
 ************************************************/

function showArk_gnb() {
  if(  $(queryId_gnb).val() != ""){
      viewGnb();
      $("#preview_wrap").show();
    }
}
/************************************************
 * 자동완성 목록을 화면에서 감춤
 * @name hideArk
 ************************************************/
function hideArk_gnb() {
    hideGnb('search');
    $("#preview_wrap").hide();
}

function viewWordRecent(){
    $(".recent_searches").show();
    $(".recent_popular_keyword").show();
}
