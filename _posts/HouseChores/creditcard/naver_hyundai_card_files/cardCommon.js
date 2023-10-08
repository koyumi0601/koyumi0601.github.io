
/*
 ***********************************************
 * @source         : cardCommon.js
 * @description    : 카드 공통
 ***********************************************
 * DATE         AUTHOR    DESCRIPTION
 * ---------------------------------------------
 * 2015.10.22   GGBK12    리뉴얼
 ***********************************************
*/

/****************************************************************************************************
 * 공통부
 ****************************************************************************************************/
// 세로플레이트카드 목록
// GGCP31 newPlateArr 변수 newPlateArr.js 로 이동

//카드공통함수
var cardCommonFn = {
    createFavoriteCard  : function() {}, // cookie에 있는 관심카드 생성
    addFavoriteCard     : function() {}, // 관심카드 추가
    delFavoriteCard     : function() {}, // 관심카드 삭제
    moveUrl             : function() {}, // 페이지 이동
    detailCard          : function() {}, // 카드 상세보기 이동
    compareCard         : function() {}  // 카드 비교 이동
};

/***************************************************************************************************
 * 함수명: cardCommonFn.createFavoriteCard
 * 설명: Cookie에 있는 관심카드 생성
 ***************************************************************************************************/
cardCommonFn.createFavoriteCard = function() {
    $base = $('.area_location .compare_link, .area_side .box_favorite_side'),
    $box = $('.favorite_card_side'),
    $txt = $base.find('h2 span, .link_open .num');

    // 관심카드함 삭제 이벤트
    $(document).on('click', '.favorite_card_side button.btn_del', function(ev) {
        ev.preventDefault();

        cardCommonFn.delFavoriteCard($(ev.target).closest('button').val());
        return false;
    });

    var cardlist = $.cookie("cardlist") || null;
    var refreshTarget = [];

    // 관심카드 박스 요소 생성
    if (cardlist != null)
    {
        $box.find('.none').remove();

        var cardCd = cardlist.split('|');
        for ( var j = 0, len = cardCd.length; j < len; j++)
        {
			if($.inArray(cardCd[j], newPlateArr) >= 0) {
				var _data = '<li class="new_plate">'
				+ '<img src="/img/com/card/card_' + cardCd[j] + '_h.png" alt="' + "" + '" />'
				+ '<span class="txt">' + "" + '</span>'
			 	+ '<button class="btn_del" value="' + cardCd[j] + '"><span>삭제</span></button>'
				+ '</li>';
				$base.addClass('hasNewPlate');
			} else {
				var _data = '<li>'
				+ '<img src="/img/com/card/card_' + cardCd[j] + '.png" alt="' + "" + '" />'
				+ '<span class="txt">' + "" + '</span>'
				+ '<button class="btn_del" value="' + cardCd[j] + '"><span>삭제</span></button>'
				+ '</li>';
			}

            $box.append(_data);
            $('#favoriteCard_' + cardCd[j]).addClass('add');
            if ( $('#favoriteCard_' + cardCd[j]).find('span').text() == '관심카드 담기' )
            {
                $('#favoriteCard_' + cardCd[j]).find('span').text('관심카드 빼기');
            }

            refreshTarget[refreshTarget.length] = cardCd[j];
        }
        setFavoriteText();
    }
    $box.find('img').css({ top : '0' });

    //ie8 리플래쉬 버그
    setTimeout(function(){
        for (var i=0, len = refreshTarget.length; i<len; i++)
        {
            $('#favoriteCard_' + refreshTarget[i]).parent().addClass('tmpClass').removeClass('tmpClass').trigger('resize');
        }
    }, 100);
};

/***************************************************************************************************
 * 함수명: cardCommonFn.addFavoriteCard
 * 설명: 관심카드 추가 (cardCd:카드코드,cardNm:카드명, 클릭엘리먼트)
 ***************************************************************************************************/
cardCommonFn.addFavoriteCard = function(cardCd, cardNm, target) {
    var $base = $('.area_location .compare_link, .area_side .box_favorite_side'),
    $box = $('.favorite_card_side'),
    $txt = $base.find('h2 span, .link_open .num');
    var cardCnt = 0;
    var cardlist = $.cookie('cardlist') || null;

    if (cardlist != null)
    {
        var card = cardlist.split('|');
        cardCnt = card.length;

        if (card.length >= 3)
        {
            alert('카드담기는 3개까지 가능합니다.');
            return false;
        }

        for ( var j = 0; j < card.length; j++)
        {
            if (card[j] == cardCd) {
                alert('이미 담겨진 카드입니다.');
                return false;
            }
        }
    }
    cardCnt++;

    $.cookie.raw = true;
    $.cookie('cardlist', ((cardlist == null) ? '' : cardlist + '|') + cardCd, {path: '/'});

	if($.inArray(cardCd, newPlateArr) >= 0) {
		var _data = '<li class="new_plate">'
		+ '<img src="/img/com/card/card_' + cardCd + '_h.png" alt="' + cardNm + '" />'
		+ '<span class="txt">' + cardNm + '</span>'
        + '<button class="btn_del" value="' + cardCd + '"><span>삭제</span></button>'
        + '</li>';
		$base.addClass('hasNewPlate');
	} else {
		var _data = '<li>'
        + '<img src="/img/com/card/card_' + cardCd + '.png" alt="' + cardNm + '" />'
		+ '<span class="txt">' + cardNm + '</span>'
        + '<button class="btn_del" value="' + cardCd + '"><span>삭제</span></button>'
        + '</li>';
	}

    $base.addClass('on');
    $box.find('.none').remove();
    $box.append(_data).find('img').css({ top : '0' });
    setFavoriteText();

    $('#' + target).removeClass('add').addClass('add'); // 관심카드에 담긴 버튼은 '-' 로 변경 처리

	setTimeout(function() {
		setTopBtnPosition()
	},300)
};

/***************************************************************************************************
 * 함수명: cardCommonFn.delFavoriteCard
 * 설명 : 관심카드 삭제
 ***************************************************************************************************/
cardCommonFn.delFavoriteCard = function(cardCd, objId) {
    if(!$('.compare_link').hasClass('on'))
    {
        $('.compare_link').addClass('on');
    }

    var $this = $('.favorite_card_side button[value=' + cardCd + ']');
    if (objId != undefined && objId != null)
    {
        $('#' + objId).removeClass('add');
    }
    else
    {
        // 페이지내 동일 카드 찾아 '-' => '+' 버튼으로 변경
        var _src = $this.siblings('img').attr('src');
        $('.box_card .img_card img').filter(function(){
            //if ($(this).attr('src') == _src){
			if ($(this).attr('src').indexOf(_src) >= 0){
                $(this).closest('.img_card').siblings('.link_favorite').find('a').removeClass('add');
            }
        });
    }

    $this.parent().find('img').animate({ top : '50px' }, 0 ,function(){
        $this.parent().remove();
        setFavoriteText();
    });

    var cardlist = $.cookie('cardlist') || null;
    var newCardList = '';

    if (cardlist != null) {
        var newCardCnt = 0;
        var card = cardlist.split('|');
        for ( var j = 0; j < card.length; j++) {
            if (card[j] != cardCd) {
                newCardCnt++;
                newCardList = ((newCardList == '') ? '' : newCardList + '|') + card[j];
            }
        }

        if (newCardCnt == 0) {
            $.removeCookie('cardlist', { expires: 0, path: '/' });
            setTimeout(function(){
                $('.compare_link').removeClass('on');
            },1500);
        } else {
            $.cookie.raw = true;
            $.cookie('cardlist', newCardList, {path: '/'});
        }
    }
	setFavoriteText();
};

function setFavoriteText(){
    var $base = $('.area_location .compare_link, .area_side .box_favorite_side'),
    $box = $('.favorite_card_side'),
    $txt = $base.find('h2 span, .link_open .num');
    var _size = $txt.size(),
        _cardSize = $box.find('li').size();
    if (_cardSize > 0){
        $txt.css({ display : 'inline-block' });
		$box.removeClass('empty');
		if($box.find('li.new_plate').length > 0) {
			$base.addClass('hasNewPlate');
		} else {
			$base.removeClass('hasNewPlate');
		}
    } else {
        $txt.css({ display : 'none' });
		$box.addClass('empty');
		$base.removeClass('hasNewPlate');
    }
    for (var i = 0; i < _size; i++){
        var _class = $txt.eq(i).attr('class'),
            _txt = '';
        if (_class == 'num'){
            _txt = _cardSize;
        } else {
            _txt = '(' + _cardSize + ')';
        }
        $txt.eq(i).text(_txt);
    }
}

/***************************************************************************************************
 * 함수명: cardCommonFn.moveUrl
 * 설명: 페이지 이동
 ***************************************************************************************************/
cardCommonFn.moveUrl = function(url){
    document.location.href = url;
};

/*******************************************************************************
 * 함수명: cardCommonFn.detailCard
 * 설명: 카드 상세보기로이동(cardflag:카드종류코드,cardWcd:카드코드,eventCode:이벤트코드) 설명
 ******************************************************************************/
 cardCommonFn.detailCard = function(cardflag, cardWcd, eventCode) {
     document.location.href = '/cpc/cr/CPCCR0201_01.hc?cardflag=' + cardflag + '&cardWcd=' + cardWcd + '&eventCode=' + eventCode;
 };

 /***************************************************************************************************
  * 함수명: cardCommonFn.compareCard
  *  설명: 카드 비교 이동
  ***************************************************************************************************/
  cardCommonFn.compareCard = function(cardflag, cardWcd) {

    var cardlist = $.cookie('cardlist') || '';

    if (cardlist != '') {
        if (cardlist.split('|').length < 2) {
            alert('관심카드가  2개 이상일때만 카드비교가 가능합니다.');
            return false;
        }
    } else {
        alert('관심카드가  2개 이상일때만 카드비교가 가능합니다.');
        return false;
    }

    document.location.href = '/cpc/cr/CPCCR0211_11.hc';
};
