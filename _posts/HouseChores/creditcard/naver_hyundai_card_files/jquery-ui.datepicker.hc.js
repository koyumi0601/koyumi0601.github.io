/* eslint-disable max-len, camelcase */
/*!
 * jQuery UI DatepickerHC @VERSION
 * http://jqueryui.com
 *
 * Copyright jQuery Foundation and other contributors
 * Released under the MIT license.
 * http://jquery.org/license
 */

//>>label: DatepickerHC
//>>group: Widgets
//>>description: Displays a calendar from an input or inline for selecting dates.
//>>docs: http://api.jqueryui.com/datepickerHC/
//>>demos: http://jqueryui.com/datepickerHC/
//>>css.structure: ../../themes/base/core.css
//>>css.structure: ../../themes/base/datepickerHC.css
//>>css.theme: ../../themes/base/theme.css

( function( factory ) {
	"use strict";

	if ( typeof define === "function" && define.amd ) {

		// AMD. Register as an anonymous module.
		define( [
			"jquery",
			"../version",
			"../keycode"
		], factory );
	} else {

		// Browser globals
		factory( jQuery );
	}
} )( function( $ ) {
"use strict";

$.extend( $.ui, { datepickerHC: { version: "@VERSION" } } );

var datepickerHC_instActive;

function datepickerHC_getZindex( elem ) {
	var position, value;
	while ( elem.length && elem[ 0 ] !== document ) {

		// Ignore z-index if position is set to a value where z-index is ignored by the browser
		// This makes behavior of this function consistent across browsers
		// WebKit always returns auto if the element is positioned
		position = elem.css( "position" );
		if ( position === "absolute" || position === "relative" || position === "fixed" ) {

			// IE returns 0 when zIndex is not specified
			// other browsers return a string
			// we ignore the case of nested elements with an explicit value of 0
			// <div style="z-index: -10;"><div style="z-index: 0;"></div></div>
			value = parseInt( elem.css( "zIndex" ), 10 );
			if ( !isNaN( value ) && value !== 0 ) {
				return value;
			}
		}
		elem = elem.parent();
	}

	return 0;
}

/* Date picker manager.
   Use the singleton instance of this class, $.datepickerHC, to interact with the date picker.
   Settings for (groups of) date pickers are maintained in an instance object,
   allowing multiple different settings on the same page. */

function DatepickerHC() {
	this._curInst = null; // The current instance in use
	this._keyEvent = true; // If the last event was a key event
	this._disabledInputs = []; // List of date picker inputs that have been disabled
	this._datepickerShowing = false; // True if the popup picker is showing , false if not
	this._inDialog = false; // True if showing within a "dialog", false if not
	this._mainDivId = "ui-datepicker-hc-div"; // The ID of the main datepickerHC division
	this._inlineClass = "ui-datepicker-hc-inline"; // The name of the inline marker class
	this._appendClass = "ui-datepicker-hc-append"; // The name of the append marker class
	this._triggerClass = "ui-datepicker-hc-trigger"; // The name of the trigger marker class
	this._dialogClass = "ui-datepicker-hc-dialog"; // The name of the dialog marker class
	this._disableClass = "ui-datepicker-hc-disabled"; // The name of the disabled covering marker class
	this._unselectableClass = "ui-datepicker-hc-unselectable"; // The name of the unselectable cell marker class
	this._hoverableClass = "ui-datepicker-hc-hoverable"; // hc-ui 추가
	this._currentClass = "ui-datepicker-hc-current-day"; // The name of the current day marker class
	this._dayOverClass = "ui-datepicker-hc-days-cell-over"; // The name of the day hover marker class
	this.regional = []; // Available regional settings, indexed by language code
	this.regional[ "en" ] = { // Default regional settings
		closeText: "Done", // Display text for close link
		prevText: "Prev", // Display text for previous month link
		nextText: "Next", // Display text for next month link
		currentText: "Today", // Display text for current month link
		monthNames: [ "January", "February", "March", "April", "May", "June",
			"July", "August", "September", "October", "November", "December" ], // Names of months for drop-down and formatting
		monthNamesShort: [ "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" ], // For formatting
		dayNames: [ "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" ], // For formatting
		dayNamesShort: [ "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" ], // For formatting
		dayNamesMin: [ "Su", "Mo", "Tu", "We", "Th", "Fr", "Sa" ], // Column headings for days starting at Sunday
		weekHeader: "Wk", // Column header for week of the year
		dateFormat: "mm/dd/yy", // See format options on parseDate
		firstDay: 0, // The first day of the week, Sun = 0, Mon = 1, ...
		isRTL: false, // True if right-to-left language, false if left-to-right
		showMonthAfterYear: false, // True if the year select precedes month, false for month then year
		yearSuffix: "", // Additional text to append to the year in the month headers,
		selectMonthLabel: "Select month", // Invisible label for month selector
		selectYearLabel: "Select year" // Invisible label for year selector
	};
	this.regional["" ] = { // Default regional settings
		closeText: "완료", // Display text for close link
		prevText: "이전", // Display text for previous month link
		nextText: "다음", // Display text for next month link
		currentText: "오늘", // Display text for current month link
		monthNames: [ "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12" ], // Names of months for drop-down and formatting
		monthNamesShort: [ "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12" ], // For formatting
		dayNames: [ "일요일", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일" ], // For formatting
		dayNamesShort: [ "일", "월", "화", "수", "목", "금", "토" ], // For formatting
		dayNamesMin: [ "일", "월", "화", "수", "목", "금", "토" ], // Column headings for days starting at Sunday
		weekHeader: "Wk", // Column header for week of the year
		dateFormat: "y. m. d", // See format options on parseDate
		firstDay: 0, // The first day of the week, Sun = 0, Mon = 1, ...
		isRTL: false, // True if right-to-left language, false if left-to-right
		showMonthAfterYear: true, // True if the year select precedes month, false for month then year
		yearSuffix: ".", // Additional text to append to the year in the month headers,
		selectMonthLabel: "월 선택", // Invisible label for month selector
		selectYearLabel: "년 선택" // Invisible label for year selector
	};
	this._defaults = { // Global defaults for all the date picker instances
		showOn: "focus", // "focus" for popup on focus,
			// "button" for trigger button, or "both" for either
		showAnim: "fadeIn", // Name of jQuery animation for popup
		showOptions: {}, // Options for enhanced animations
		defaultDate: null, // Used when field is blank: actual date,
			// +/-number for offset from today, null for today
		appendText: "", // Display text following the input box, e.g. showing the format
		buttonText: "...", // Text for trigger button
		buttonImage: "", // URL for trigger button image
		buttonImageOnly: false, // True if the image appears alone, false if it appears on a button
		hideIfNoPrevNext: false, // True to hide next/previous month links
			// if not applicable, false to just disable them
		navigationAsDateFormat: false, // True if date formatting applied to prev/today/next links
		gotoCurrent: false, // True if today link goes back to current selection instead
		changeMonth: false, // True if month can be selected directly, false if only prev/next
		changeYear: false, // True if year can be selected directly, false if only prev/next
		yearRange: "c-10:c+10", // Range of years to display in drop-down,
			// either relative to today's year (-nn:+nn), relative to currently displayed year
			// (c-nn:c+nn), absolute (nnnn:nnnn), or a combination of the above (nnnn:-n)
		showOtherMonths: false, // True to show dates in other months, false to leave blank
		selectOtherMonths: false, // True to allow selection of dates in other months, false for unselectable
		showWeek: false, // True to show week of the year, false to not show it
		calculateWeek: this.iso8601Week, // How to calculate the week of the year,
			// takes a Date and returns the number of the week for it
		shortYearCutoff: "+10", // Short year values < this are in the current century,
			// > this are in the previous century,
			// string value starting with "+" for current year + value
		//selectedDateLength: 0,
		minSelectableDate: null, // hc-ui : mindate와 별개로 최소 선택 가능한 날짜 정의
		minDate: null, // The earliest selectable date, or null for no limit
		maxDate: null, // The latest selectable date, or null for no limit
		duration: "fast", // Duration of display/closure
		beforeShowDay: null, // Function that takes a date and returns an array with
			// [0] = true if selectable, false if not, [1] = custom CSS class name(s) or "",
			// [2] = cell title (optional), e.g. $.datepickerHC.noWeekends
		beforeShow: null, // Function that takes an input field and
			// returns a set of custom settings for the date picker
		onSelect: null, // Define a callback function when a date is selected
		onChangeMonthYear: null, // Define a callback function when the month or year is changed
		onClose: null, // Define a callback function when the datepickerHC is closed
		onUpdateDatepicker: null, // Define a callback function when the datepickerHC is updated
		numberOfMonths: 1, // Number of months to show at a time
		showCurrentAtPos: 0, // The position in multipe months at which to show the current month (starting at 0)
		stepMonths: 1, // Number of months to step back/forward
		stepBigMonths: 12, // Number of months to step back/forward for the big links
		altField: "", // Selector for an alternate field to store selected dates into
		altFormat: "", // The date format to use for the alternate field
		constrainInput: true, // The input is constrained by the current date format
		showButtonPanel: false, // True to show button panel, false to not show it
		autoSize: false, // True to size the input for the date format, false to leave as is
		disabled: false, // The initial disabled state
		speedScroll: 500, //hc-ui : 캘린더간 버튼으로 이동 시 스크롤 스피드 기본값(ms)
		rangeOneCal: false, //hc-ui : 1개의 캘린더로 from ~ to 설정하는 케이스의 옵션 기본값 false
		idRangeTo: null //hc-ui : 1개의 캘린더로 from ~ to 설정하는 케이스일때 to값 필드 id
	};
	$.extend( this._defaults, this.regional[ "" ] );
	this.regional.en = $.extend( true, {}, this.regional[ "" ] );
	this.regional[ "en-US" ] = $.extend( true, {}, this.regional.en );
	this.dpDiv = datepickerHC_bindHover( $( "<div id='" + this._mainDivId + "' class='ui-datepicker-hc ui-widget ui-widget-content ui-helper-clearfix ui-corner-all'></div>" ) );
}

$.extend( DatepickerHC.prototype, {

	/* Class name added to elements to indicate already configured with a date picker. */
	markerClassName: "hasDatepickerHC",

	//Keep track of the maximum number of rows displayed (see #7043)
	maxRows: 4,

	//selectedDateLength:0, //ui-hc : 1캘린더로 범위 설정을 위해 선택 날짜 갯수 저장용 변수 선언

	// TODO rename to "widget" when switching to widget factory
	_widgetDatepickerHC: function() {
		return this.dpDiv;
	},

	/* Override the default settings for all instances of the date picker.
	 * @param  settings  object - the new settings to use as defaults (anonymous object)
	 * @return the manager object
	 */
	setDefaults: function( settings ) {
		datepickerHC_extendRemove( this._defaults, settings || {} );
		return this;
	},

	/* Attach the date picker to a jQuery selection.
	 * @param  target	element - the target input field or division or span
	 * @param  settings  object - the new settings to use for this date picker instance (anonymous)
	 */
	_attachDatepickerHC: function( target, settings ) {
		var nodeName, inline, inst;
		nodeName = target.nodeName.toLowerCase();
		inline = ( nodeName === "div" || nodeName === "span" );
		if ( !target.id ) {
			this.uuid += 1;
			target.id = "dp" + this.uuid;
		}
		inst = this._newInst( $( target ), inline );
		inst.settings = $.extend( {}, settings || {} );
		if ( nodeName === "input" ) {
			this._connectDatepickerHC( target, inst );
		} else if ( inline ) {
			this._inlineDatepickerHC( target, inst );
		}
	},

	/* Create a new instance object. */
	_newInst: function( target, inline ) {
		var id = target[ 0 ].id.replace( /([^A-Za-z0-9_\-])/g, "\\\\$1" ); // escape jQuery meta chars
		return { id: id, input: target, // associated target
			selectedDay: 0, selectedMonth: 0, selectedYear: 0, // current selection
			drawMonth: 0, drawYear: 0, // month being drawn
			inline: inline, // is datepickerHC inline or not
			dpDiv: ( !inline ? this.dpDiv : // presentation div
			datepickerHC_bindHover( $( "<div class='" + this._inlineClass + " ui-datepicker-hc ui-widget ui-widget-content ui-helper-clearfix ui-corner-all'></div>" ) ) ) };
	},

	/* Attach the date picker to an input field. */
	_connectDatepickerHC: function( target, inst ) {
		//console.log("_connectDatepickerHC")
		var input = $( target );
		inst.append = $( [] );
		inst.trigger = $( [] );
		var input2;
		if(inst.settings.idRangeTo){
			input2 = $("#" + inst.settings.idRangeTo);
		}
		if ( input.hasClass( this.markerClassName ) ) {
			return;
		}
		this._attachments( input, inst );
		input.addClass( this.markerClassName ).on( "keydown", this._doKeyDown ).
			on( "keypress", this._doKeyPress ).on( "keyup", this._doKeyUp );
		if(inst.settings.idRangeTo){
			input2.addClass( this.markerClassName ).attr("data-from-id", target.id).attr("data-selected-year","").attr("data-selected-month","").attr("data-selected-day","").on( "keypress", this._doKeyPress ).on( "keyup", this._doKeyUp );
		}
		this._autoSize( inst );
		$.data( target, "datepickerHC", inst );
		//If disabled option is true, disable the datepickerHC once it has been attached to the input (see ticket #5665)
		if ( inst.settings.disabled ) {
			this._disableDatepickerHC( target );
		}
	},

	/* Make attachments based on settings. */
	_attachments: function( input, inst ) {
		var showOn, buttonText, buttonImage,
			appendText = this._get( inst, "appendText" ),
			isRTL = this._get( inst, "isRTL" );
		
		//hc-ui : 1캘린더 range select 케이스일 경우 2번째 필드 정의
		var input2 = null;
		if(inst.settings.idRangeTo) {
			$("#" + inst.id).attr("data-selected-length",0);
			input2 = $("#" + inst.settings.idRangeTo);
		}

		if ( inst.append ) {
			inst.append.remove();
		}
		if ( appendText ) {
			inst.append = $( "<span>" )
				.addClass( this._appendClass )
				.text( appendText );
			input[ isRTL ? "before" : "after" ]( inst.append );
		}

		input.off( "focus", this._showDatepickerHC );

		if ( inst.trigger ) {
			inst.trigger.remove();
		}

		showOn = this._get( inst, "showOn" );
		if ( showOn === "focus" || showOn === "both" ) { // pop-up date picker when in the marked field
			input.on( "focus", this._showDatepickerHC );
			var _showDPHC =  this._showDatepickerHC;
			if(inst.settings.idRangeTo) {
				input2.on("focus", function(){
					$.datepickerHC._showDatepickerHC(input[ 0 ], input2)
				});
				//input2.on( "focus", _showDPHC);
				//input2.on( "focus", $.datepickerHC._showDatepickerHC( input[ 0 ] ));
			}
		}
		if ( showOn === "button" || showOn === "both" ) { // pop-up date picker when button clicked
			buttonText = this._get( inst, "buttonText" );
			buttonImage = this._get( inst, "buttonImage" );

			if ( this._get( inst, "buttonImageOnly" ) ) {
				inst.trigger = $( "<img>" )
					.addClass( this._triggerClass )
					.attr( {
						src: buttonImage,
						alt: buttonText,
						title: buttonText
					} );
			} else {
				inst.trigger = $( "<button type='button'>" )
					.addClass( this._triggerClass );
				if ( buttonImage ) {
					inst.trigger.html(
						$( "<img>" )
							.attr( {
								src: buttonImage,
								alt: buttonText,
								title: buttonText
							} )
					);
				} else {
					inst.trigger.text( buttonText );
				}
			}

			input[ isRTL ? "before" : "after" ]( inst.trigger );
			inst.trigger.on( "click", function() {
				if ( $.datepickerHC._datepickerShowing && $.datepickerHC._lastInput === input[ 0 ] ) {
					$.datepickerHC._hideDatepickerHC();
				} else if ( $.datepickerHC._datepickerShowing && $.datepickerHC._lastInput !== input[ 0 ] ) {
					$.datepickerHC._hideDatepickerHC();
					$.datepickerHC._showDatepickerHC( input[ 0 ] );
				} else {
					$.datepickerHC._showDatepickerHC( input[ 0 ] );
				}
				return false;
			} );
		}
	},

	/* Apply the maximum length for the date format. */
	_autoSize: function( inst ) {
		if ( this._get( inst, "autoSize" ) && !inst.inline ) {
			var findMax, max, maxI, i,
				date = new Date( 2009, 12 - 1, 20 ), // Ensure double digits
				dateFormat = this._get( inst, "dateFormat" );

			if ( dateFormat.match( /[DM]/ ) ) {
				findMax = function( names ) {
					max = 0;
					maxI = 0;
					for ( i = 0; i < names.length; i++ ) {
						if ( names[ i ].length > max ) {
							max = names[ i ].length;
							maxI = i;
						}
					}
					return maxI;
				};
				date.setMonth( findMax( this._get( inst, ( dateFormat.match( /MM/ ) ?
					"monthNames" : "monthNamesShort" ) ) ) );
				date.setDate( findMax( this._get( inst, ( dateFormat.match( /DD/ ) ?
					"dayNames" : "dayNamesShort" ) ) ) + 20 - date.getDay() );
			}
			inst.input.attr( "size", this._formatDate( inst, date ).length );
		}
	},

	/* Attach an inline date picker to a div. */
	_inlineDatepickerHC: function( target, inst ) {
		var divSpan = $( target );
		if ( divSpan.hasClass( this.markerClassName ) ) {
			return;
		}
		divSpan.addClass( this.markerClassName ).append( inst.dpDiv );
		$.data( target, "datepickerHC", inst );
		this._setDate( inst, this._getDefaultDate( inst ), true );
		this._updateDatepickerHC( inst );
		this._updateAlternate( inst );

		//If disabled option is true, disable the datepickerHC before showing it (see ticket #5665)
		if ( inst.settings.disabled ) {
			this._disableDatepickerHC( target );
		}

		// Set display:block in place of inst.dpDiv.show() which won't work on disconnected elements
		// http://bugs.jqueryui.com/ticket/7552 - A DatepickerHC created on a detached div has zero height
		inst.dpDiv.css( "display", "block" );
	},

	/* Pop-up the date picker in a "dialog" box.
	 * @param  input element - ignored
	 * @param  date	string or Date - the initial date to display
	 * @param  onSelect  function - the function to call when a date is selected
	 * @param  settings  object - update the dialog date picker instance's settings (anonymous object)
	 * @param  pos int[2] - coordinates for the dialog's position within the screen or
	 *					event - with x/y coordinates or
	 *					leave empty for default (screen centre)
	 * @return the manager object
	 */
	_dialogDatepickerHC: function( input, date, onSelect, settings, pos ) {
		var id, browserWidth, browserHeight, scrollX, scrollY,
			inst = this._dialogInst; // internal instance

		if ( !inst ) {
			this.uuid += 1;
			id = "dp" + this.uuid;
			this._dialogInput = $( "<input type='text' id='" + id +
				"' style='position: absolute; top: -100px; width: 0px;'/>" );
			this._dialogInput.on( "keydown", this._doKeyDown );
			$( "body" ).append( this._dialogInput );
			inst = this._dialogInst = this._newInst( this._dialogInput, false );
			inst.settings = {};
			$.data( this._dialogInput[ 0 ], "datepickerHC", inst );
		}
		datepickerHC_extendRemove( inst.settings, settings || {} );
		date = ( date && date.constructor === Date ? this._formatDate( inst, date ) : date );
		this._dialogInput.val( date );

		this._pos = ( pos ? ( pos.length ? pos : [ pos.pageX, pos.pageY ] ) : null );
		if ( !this._pos ) {
			browserWidth = document.documentElement.clientWidth;
			browserHeight = document.documentElement.clientHeight;
			scrollX = document.documentElement.scrollLeft || document.body.scrollLeft;
			scrollY = document.documentElement.scrollTop || document.body.scrollTop;
			this._pos = // should use actual width/height below
				[ ( browserWidth / 2 ) - 100 + scrollX, ( browserHeight / 2 ) - 150 + scrollY ];
		}

		// Move input on screen for focus, but hidden behind dialog
		this._dialogInput.css( "left", ( this._pos[ 0 ] + 20 ) + "px" ).css( "top", this._pos[ 1 ] + "px" );
		inst.settings.onSelect = onSelect;
		this._inDialog = true;
		this.dpDiv.addClass( this._dialogClass );
		this._showDatepickerHC( this._dialogInput[ 0 ] );
		if ( $.blockUI ) {
			$.blockUI( this.dpDiv );
		}
		$.data( this._dialogInput[ 0 ], "datepickerHC", inst );
		return this;
	},

	/* Detach a datepickerHC from its control.
	 * @param  target	element - the target input field or division or span
	 */
	_destroyDatepickerHC: function( target ) {
		var nodeName,
			$target = $( target ),
			inst = $.data( target, "datepickerHC" );

		if ( !$target.hasClass( this.markerClassName ) ) {
			return;
		}

		nodeName = target.nodeName.toLowerCase();
		$.removeData( target, "datepickerHC" );
		if ( nodeName === "input" ) {
			inst.append.remove();
			inst.trigger.remove();
			$target.removeClass( this.markerClassName ).
				off( "focus", this._showDatepickerHC ).
				off( "keydown", this._doKeyDown ).
				off( "keypress", this._doKeyPress ).
				off( "keyup", this._doKeyUp );
		} else if ( nodeName === "div" || nodeName === "span" ) {
			$target.removeClass( this.markerClassName ).empty();
		}

		if ( datepickerHC_instActive === inst ) {
			datepickerHC_instActive = null;
			this._curInst = null;
		}
	},

	/* Enable the date picker to a jQuery selection.
	 * @param  target	element - the target input field or division or span
	 */
	_enableDatepickerHC: function( target ) {
		var nodeName, inline,
			$target = $( target ),
			inst = $.data( target, "datepickerHC" );

		if ( !$target.hasClass( this.markerClassName ) ) {
			return;
		}

		nodeName = target.nodeName.toLowerCase();
		if ( nodeName === "input" ) {
			target.disabled = false;
			inst.trigger.filter( "button" ).
				each( function() {
					this.disabled = false;
				} ).end().
				filter( "img" ).css( { opacity: "1.0", cursor: "" } );
		} else if ( nodeName === "div" || nodeName === "span" ) {
			inline = $target.children( "." + this._inlineClass );
			inline.children().removeClass( "ui-state-disabled" );
			inline.find( "select.ui-datepicker-hc-month, select.ui-datepicker-hc-year" ).
				prop( "disabled", false );
		}
		this._disabledInputs = $.map( this._disabledInputs,

			// Delete entry
			function( value ) {
				return ( value === target ? null : value );
			} );
	},

	/* Disable the date picker to a jQuery selection.
	 * @param  target	element - the target input field or division or span
	 */
	_disableDatepickerHC: function( target ) {
		var nodeName, inline,
			$target = $( target ),
			inst = $.data( target, "datepickerHC" );

		if ( !$target.hasClass( this.markerClassName ) ) {
			return;
		}

		nodeName = target.nodeName.toLowerCase();
		if ( nodeName === "input" ) {
			target.disabled = true;
			inst.trigger.filter( "button" ).
				each( function() {
					this.disabled = true;
				} ).end().
				filter( "img" ).css( { opacity: "0.5", cursor: "default" } );
		} else if ( nodeName === "div" || nodeName === "span" ) {
			inline = $target.children( "." + this._inlineClass );
			inline.children().addClass( "ui-state-disabled" );
			inline.find( "select.ui-datepicker-hc-month, select.ui-datepicker-hc-year" ).
				prop( "disabled", true );
		}
		this._disabledInputs = $.map( this._disabledInputs,

			// Delete entry
			function( value ) {
				return ( value === target ? null : value );
			} );
		this._disabledInputs[ this._disabledInputs.length ] = target;
	},

	/* Is the first field in a jQuery collection disabled as a datepickerHC?
	 * @param  target	element - the target input field or division or span
	 * @return boolean - true if disabled, false if enabled
	 */
	_isDisabledDatepickerHC: function( target ) {
		if ( !target ) {
			return false;
		}
		for ( var i = 0; i < this._disabledInputs.length; i++ ) {
			if ( this._disabledInputs[ i ] === target ) {
				return true;
			}
		}
		return false;
	},

	/* Retrieve the instance data for the target control.
	 * @param  target  element - the target input field or division or span
	 * @return  object - the associated instance data
	 * @throws  error if a jQuery problem getting data
	 */
	_getInst: function( target ) {
		try {
			return $.data( target, "datepickerHC" );
		} catch ( err ) {
			throw "Missing instance data for this datepickerHC";
		}
	},

	/* Update or retrieve the settings for a date picker attached to an input field or division.
	 * @param  target  element - the target input field or division or span
	 * @param  name	object - the new settings to update or
	 *				string - the name of the setting to change or retrieve,
	 *				when retrieving also "all" for all instance settings or
	 *				"defaults" for all global defaults
	 * @param  value   any - the new value for the setting
	 *				(omit if above is an object or to retrieve a value)
	 */
	_optionDatepickerHC: function( target, name, value ) {
		var settings, date, minDate, maxDate,
			inst = this._getInst( target );

		if ( arguments.length === 2 && typeof name === "string" ) {
			return ( name === "defaults" ? $.extend( {}, $.datepickerHC._defaults ) :
				( inst ? ( name === "all" ? $.extend( {}, inst.settings ) :
				this._get( inst, name ) ) : null ) );
		}

		settings = name || {};
		if ( typeof name === "string" ) {
			settings = {};
			settings[ name ] = value;
		}

		if ( inst ) {
			if ( this._curInst === inst ) {
				this._hideDatepickerHC();
			}

			date = this._getDateDatepickerHC( target, true );
			minDate = this._getMinMaxDate( inst, "min" );
			maxDate = this._getMinMaxDate( inst, "max" );
			datepickerHC_extendRemove( inst.settings, settings );

			// reformat the old minDate/maxDate values if dateFormat changes and a new minDate/maxDate isn't provided
			if ( minDate !== null && settings.dateFormat !== undefined && settings.minDate === undefined ) {
				inst.settings.minDate = this._formatDate( inst, minDate );
			}
			if ( maxDate !== null && settings.dateFormat !== undefined && settings.maxDate === undefined ) {
				inst.settings.maxDate = this._formatDate( inst, maxDate );
			}
			if ( "disabled" in settings ) {
				if ( settings.disabled ) {
					this._disableDatepickerHC( target );
				} else {
					this._enableDatepickerHC( target );
				}
			}
			this._attachments( $( target ), inst );
			this._autoSize( inst );
			this._setDate( inst, date );
			this._updateAlternate( inst );
			this._updateDatepickerHC( inst );
		}
	},

	// Change method deprecated
	_changeDatepickerHC: function( target, name, value ) {
		this._optionDatepickerHC( target, name, value );
	},

	/* Redraw the date picker attached to an input field or division.
	 * @param  target  element - the target input field or division or span
	 */
	_refreshDatepickerHC: function( target ) {
		var inst = this._getInst( target );
		if ( inst ) {
			this._updateDatepickerHC( inst );
		}
	},

	/* Set the dates for a jQuery selection.
	 * @param  target element - the target input field or division or span
	 * @param  date	Date - the new date
	 */
	_setDateDatepickerHC: function( target, date ) {
		var inst = this._getInst( target );
		if ( inst ) {
			this._setDate( inst, date );
			this._updateDatepickerHC( inst );
			this._updateAlternate( inst );
		}
	},

	/* Get the date(s) for the first entry in a jQuery selection.
	 * @param  target element - the target input field or division or span
	 * @param  noDefault boolean - true if no default date is to be used
	 * @return Date - the current date
	 */
	_getDateDatepickerHC: function( target, noDefault ) {
		var inst = this._getInst( target );
		if ( inst && !inst.inline ) {
			this._setDateFromField( inst, noDefault );
		}
		return ( inst ? this._getDate( inst ) : null );
	},

	/* Handle keystrokes. */
	_doKeyDown: function( event ) {
		var onSelect, dateStr, sel,
			inst = $.datepickerHC._getInst( event.target ),
			handled = true,
			isRTL = inst.dpDiv.is( ".ui-datepicker-hc-rtl" );

		inst._keyEvent = true;
		if ( $.datepickerHC._datepickerShowing ) {
			switch ( event.keyCode ) {
				case 9: $.datepickerHC._hideDatepickerHC();
						handled = false;
						break; // hide on tab out
				case 13: sel = $( "td." + $.datepickerHC._dayOverClass + ":not(." +
									$.datepickerHC._currentClass + ")", inst.dpDiv );
						if ( sel[ 0 ] ) {
							$.datepickerHC._selectDay( event.target, inst.selectedMonth, inst.selectedYear, sel[ 0 ] );
						}

						onSelect = $.datepickerHC._get( inst, "onSelect" );
						if ( onSelect ) {
							dateStr = $.datepickerHC._formatDate( inst );

							// Trigger custom callback
							onSelect.apply( ( inst.input ? inst.input[ 0 ] : null ), [ dateStr, inst ] );
						} else {
							$.datepickerHC._hideDatepickerHC();
						}

						return false; // don't submit the form
				case 27: $.datepickerHC._hideDatepickerHC();
						break; // hide on escape
				case 33: $.datepickerHC._adjustDate( event.target, ( event.ctrlKey ?
							-$.datepickerHC._get( inst, "stepBigMonths" ) :
							-$.datepickerHC._get( inst, "stepMonths" ) ), "M" );
						break; // previous month/year on page up/+ ctrl
				case 34: $.datepickerHC._adjustDate( event.target, ( event.ctrlKey ?
							+$.datepickerHC._get( inst, "stepBigMonths" ) :
							+$.datepickerHC._get( inst, "stepMonths" ) ), "M" );
						break; // next month/year on page down/+ ctrl
				case 35: if ( event.ctrlKey || event.metaKey ) {
							$.datepickerHC._clearDate( event.target );
						}
						handled = event.ctrlKey || event.metaKey;
						break; // clear on ctrl or command +end
				case 36: if ( event.ctrlKey || event.metaKey ) {
							$.datepickerHC._gotoToday( event.target );
						}
						handled = event.ctrlKey || event.metaKey;
						break; // current on ctrl or command +home
				case 37: if ( event.ctrlKey || event.metaKey ) {
							$.datepickerHC._adjustDate( event.target, ( isRTL ? +1 : -1 ), "D" );
						}
						handled = event.ctrlKey || event.metaKey;

						// -1 day on ctrl or command +left
						if ( event.originalEvent.altKey ) {
							$.datepickerHC._adjustDate( event.target, ( event.ctrlKey ?
								-$.datepickerHC._get( inst, "stepBigMonths" ) :
								-$.datepickerHC._get( inst, "stepMonths" ) ), "M" );
						}

						// next month/year on alt +left on Mac
						break;
				case 38: if ( event.ctrlKey || event.metaKey ) {
							$.datepickerHC._adjustDate( event.target, -7, "D" );
						}
						handled = event.ctrlKey || event.metaKey;
						break; // -1 week on ctrl or command +up
				case 39: if ( event.ctrlKey || event.metaKey ) {
							$.datepickerHC._adjustDate( event.target, ( isRTL ? -1 : +1 ), "D" );
						}
						handled = event.ctrlKey || event.metaKey;

						// +1 day on ctrl or command +right
						if ( event.originalEvent.altKey ) {
							$.datepickerHC._adjustDate( event.target, ( event.ctrlKey ?
								+$.datepickerHC._get( inst, "stepBigMonths" ) :
								+$.datepickerHC._get( inst, "stepMonths" ) ), "M" );
						}

						// next month/year on alt +right
						break;
				case 40: if ( event.ctrlKey || event.metaKey ) {
							$.datepickerHC._adjustDate( event.target, +7, "D" );
						}
						handled = event.ctrlKey || event.metaKey;
						break; // +1 week on ctrl or command +down
				default: handled = false;
			}
		} else if ( event.keyCode === 36 && event.ctrlKey ) { // display the date picker on ctrl+home
			$.datepickerHC._showDatepickerHC( this );
		} else {
			handled = false;
		}

		if ( handled ) {
			event.preventDefault();
			event.stopPropagation();
		}
	},

	/* Filter entered characters - based on date format. */
	_doKeyPress: function( event ) {
		var chars, chr,
			inst = $.datepickerHC._getInst( event.target );
		if(!inst) inst =  $.datepickerHC._curInst;
		if ( $.datepickerHC._get( inst, "constrainInput" ) ) {
			chars = $.datepickerHC._possibleChars( $.datepickerHC._get( inst, "dateFormat" ) );
			chr = String.fromCharCode( event.charCode == null ? event.keyCode : event.charCode );
			return event.ctrlKey || event.metaKey || ( chr < " " || !chars || chars.indexOf( chr ) > -1 );
		}
	},

	/* Synchronise manual entry and field/alternate field. */
	_doKeyUp: function( event ) {
		var date,
			inst = $.datepickerHC._getInst( event.target );
		if(!inst) inst =  $.datepickerHC._curInst;
		if ( inst.input.val() !== inst.lastVal ) {
			try {
				date = $.datepickerHC.parseDate( $.datepickerHC._get( inst, "dateFormat" ),
					( inst.input ? inst.input.val() : null ),
					$.datepickerHC._getFormatConfig( inst ) );

				if ( date ) { // only if valid
					$.datepickerHC._setDateFromField( inst );
					$.datepickerHC._updateAlternate( inst );
					$.datepickerHC._updateDatepickerHC( inst );
				}
			} catch ( err ) {
			}
		}
		return true;
	},

	/* Pop-up the date picker for a given input field.
	 * If false returned from beforeShow event handler do not show.
	 * @param  input  element - the input field attached to the date picker or
	 *					event - if triggered by focus
	 */
	_showDatepickerHC: function( input, input2 ) {
		input = input.target || input;
		if ( input.nodeName.toLowerCase() !== "input" ) { // find from button/image trigger
			input = $( "input", input.parentNode )[ 0 ];
		}

		if ( $.datepickerHC._isDisabledDatepickerHC( input ) || $.datepickerHC._lastInput === input ) { // already here
			return;
		}

		var inst, beforeShow, beforeShowSettings, isFixed,
			offset, showAnim, duration;

		inst = $.datepickerHC._getInst( input );
		if ( $.datepickerHC._curInst && $.datepickerHC._curInst !== inst ) {
			$.datepickerHC._curInst.dpDiv.stop( true, true );
			if ( inst && $.datepickerHC._datepickerShowing ) {
				$.datepickerHC._hideDatepickerHC( $.datepickerHC._curInst.input[ 0 ] );
			}
		}

		beforeShow = $.datepickerHC._get( inst, "beforeShow" );
		beforeShowSettings = beforeShow ? beforeShow.apply( input, [ input, inst ] ) : {};
		if ( beforeShowSettings === false ) {
			return;
		}
		datepickerHC_extendRemove( inst.settings, beforeShowSettings );

		inst.lastVal = null;
		$.datepickerHC._lastInput = input;
		$.datepickerHC._setDateFromField( inst );

		if ( $.datepickerHC._inDialog ) { // hide cursor
			input.value = "";
		}
		if ( !$.datepickerHC._pos ) { // position below input
			$.datepickerHC._pos = $.datepickerHC._findPos( input );
			$.datepickerHC._pos[ 1 ] += input.offsetHeight; // add the height
		}

		isFixed = false;
		$( input ).parents().each( function() {
			isFixed |= $( this ).css( "position" ) === "fixed";
			return !isFixed;
		} );

		offset = { left: $.datepickerHC._pos[ 0 ], top: $.datepickerHC._pos[ 1 ] };
		$.datepickerHC._pos = null;

		//to avoid flashes on Firefox
		inst.dpDiv.empty();

		// determine sizing offscreen
		inst.dpDiv.css( { position: "absolute", display: "block", top: "-1000px" } );
		$.datepickerHC._updateDatepickerHC( inst );

		// fix width for dynamic number of date pickers
		// and adjust position before showing
		offset = $.datepickerHC._checkOffset( inst, offset, isFixed );
		inst.dpDiv.css( { position: ( $.datepickerHC._inDialog && $.blockUI ?
			"static" : ( isFixed ? "fixed" : "absolute" ) ), display: "none",
			left: offset.left + "px", top: offset.top + "px" } );

		if ( !inst.inline ) {
			showAnim = $.datepickerHC._get( inst, "showAnim" );
			duration = $.datepickerHC._get( inst, "duration" );
			inst.dpDiv.css( "z-index", datepickerHC_getZindex( $( input ) ) + 1 );
			$.datepickerHC._datepickerShowing = true;

			if ( $.effects && $.effects.effect[ showAnim ] ) {
				inst.dpDiv.show( showAnim, $.datepickerHC._get( inst, "showOptions" ), duration );
			} else {
				inst.dpDiv[ showAnim || "show" ]( showAnim ? duration : null );
			}

			if ( $.datepickerHC._shouldFocusInput( inst ) ) {
				if(!input2) inst.input.trigger( "focus" ); //hc-ui : 1 캘린더 range selector 케이스에서 to 필드를 인자로 넘길때는 from 필드에 포커스 가지 않게 처리
			}

			$.datepickerHC._curInst = inst;
		}
	},

	//온 스크롤 이벤트 바인딩. 이벤트 발생 시 더미 스크롤바 포지션 처리 및 타이틀 변경 처리
	_onScroll: function(inst){
		//var scrHeight = $("#" + this._mainDivId).find(".ui-datepicker-hc-holder").height();
		var thisDiv = this._mainDivId;
		var monthHeight = $("#" + thisDiv).find(".ui-datepicker-hc-scroll").height();
		//console.log("온스크롤");

		$("#" + thisDiv + " .ui-datepicker-hc-holder").scroll( function(event){
			//console.log("스크롤");
			var _this = $(this);
			var scrRatio = _this.scrollTop() / monthHeight * 100;
			$("#" + thisDiv).attr("data-scrolltop",_this.scrollTop());
			$("#" + thisDiv + " .ui-datepicker-hc-scrollbar").css("top",scrRatio + "%"); //스크롤값을 스크롤바 포지션으로 치환
			//$("#" + thisDiv + " .ui-datepicker-hc-holder")
			$("#" + thisDiv).find(".ui-datepicker-hc-group").each(function(index, element){
				var scrT = _this.scrollTop();
				var posT = $(this).position().top;
				var outH = $(this).outerHeight()
				//console.log(scrT)
				if(posT < 0 && posT * -1 < outH){
					var currentMG = $("#" + thisDiv).find(".ui-datepicker-hc-group").eq(index);
					var currentText1 = currentMG.find(".ui-datepicker-hc-year").text();
					var currentText2 = currentMG.find(".ui-datepicker-hc-month").text();
					//console.log(currentText1)
					var currentText = currentText1 + ". " + currentText2;
					$("#" + thisDiv).find(".ui-datepicker-hc-current").text(currentText).attr("data-current-year", currentText1).attr("data-current-month", currentText2);
				}
				if(scrT < 44){
					$("#" + thisDiv).find(".ui-datepicker-hc-current").text("").attr("data-current-year", 0).attr("data-current-month", 0);
				}
			});
		});
	},

	/* Generate the date picker content. */
	_updateDatepickerHC: function( inst ) {
		//console.log("업데이트");
		//console.log(inst);
		//console.log("선택값갯수 : " + this._defaults.selectedDateLength)
		this.maxRows = 4; //Reset the max number of rows being displayed (see #7043)
		datepickerHC_instActive = inst; // for delegate hover events
		inst.dpDiv.empty().append( this._generateHTML( inst ) ).attr("data-animation", "");
		this._attachHandlers( inst );

		var origyearshtml,
			numMonths = this._getNumberOfMonths( inst ),
			cols = numMonths[ 1 ],
			width = 17,
			activeCell = inst.dpDiv.find( "." + this._dayOverClass + " a" ),
			onUpdateDatepicker = $.datepickerHC._get( inst, "onUpdateDatepicker" );

		//var selectedDateLength = 0; //hc-ui : 1캘린더로 범위 설정 시 날짜 선택 갯수 저장용 변수

		if ( activeCell.length > 0 ) {
			datepickerHC_handleMouseover.apply( activeCell.get( 0 ) );
		}

		inst.dpDiv.removeClass( "ui-datepicker-hc-multi-2 ui-datepicker-hc-multi-3 ui-datepicker-hc-multi-4" ).width( "" );
		if ( cols > 1 ) {
			inst.dpDiv.addClass( "ui-datepicker-hc-multi-" + cols ).css( "width", ( width * cols ) + "em" );
		}
		inst.dpDiv[ ( numMonths[ 0 ] !== 1 || numMonths[ 1 ] !== 1 ? "add" : "remove" ) +
			"Class" ]( "ui-datepicker-hc-multi" );
		inst.dpDiv[ ( this._get( inst, "isRTL" ) ? "add" : "remove" ) +
			"Class" ]( "ui-datepicker-hc-rtl" );

		if ( inst === $.datepickerHC._curInst && $.datepickerHC._datepickerShowing && $.datepickerHC._shouldFocusInput( inst ) ) {
			inst.input.trigger( "focus" );
		}

		// Deffered render of the years select (to avoid flashes on Firefox)
		if ( inst.yearshtml ) {
			origyearshtml = inst.yearshtml;
			setTimeout( function() {

				//assure that inst.yearshtml didn't change.
				if ( origyearshtml === inst.yearshtml && inst.yearshtml ) {
					inst.dpDiv.find( "select.ui-datepicker-hc-year" ).first().replaceWith( inst.yearshtml );
				}
				origyearshtml = inst.yearshtml = null;
			}, 0 );
		}

		if ( onUpdateDatepicker ) {
			onUpdateDatepicker.apply( ( inst.input ? inst.input[ 0 ] : null ), [ inst ] );
		}
		//hc-ui : 달력 업데이트 후 해당 영역에 스크롤바 높이 처리
		var scrHeight = $("#" + this._mainDivId).find(".ui-datepicker-hc-holder").height();
		var monthHeight = $("#" + this._mainDivId).find(".ui-datepicker-hc-scroll").height();
		var ratio = scrHeight / monthHeight * 100;
		$("#" + this._mainDivId).find(".ui-datepicker-hc-scrollbar").height(ratio + "%");

		//hc-ui : 온스크롤 함수 호출
		this._onScroll(inst);

		//hc-ui : 최초 캘린더 업데이트시 선택값이 없으면 현재 달로 스크롤, 선택 값이 있으면 해당 월로 스크롤 처리
		var thisDiv = this._mainDivId;
		//hc-ui : 선택값이 없으면 기본 값(금월,금년)으로 정의
		var goMonth = inst.currentDay == 0 ? inst.selectedMonth : inst.currentMonth;
		var goYear = inst.currentDay == 0 ? inst.selectedYear : inst.currentYear;

		//hc-ui : each문으로 선택 월/년 매칭하여 해당 달로 이동 처리
		//console.log("rangeOneCal : " + this._get( inst, "rangeOneCal" ))
		var rangeOneCal = this._get( inst, "rangeOneCal" )
		if(!rangeOneCal){
			this._gotoMonth(goYear, goMonth, false, inst);
		}else{
			var scr = inst.dpDiv.attr("data-scrolltop");
			if(scr){
				//console.log(scr)
				var scrTo = scr * 1;
				//$("#" + thisDiv).attr("data-scrolltop",_this.scrollTop());
				//console.log(inst.dpDiv.attr("data-scrolltop"))
				$("#" + this._mainDivId + " .ui-datepicker-hc-holder").scrollTop(inst.dpDiv.attr("data-scrolltop") * 1);
			}else{
				this._gotoMonth(goYear, goMonth, false, inst);
			}
			this._drawRangeHover(inst)
		}

		//hc-ui : 오늘, 이전달, 다음달 이동 버튼 처리
		var btnGoTodayMonth = $("#" + thisDiv).find(".ui-datepicker-hc-today");
		var btnGoPrevMonth = $("#" + thisDiv).find(".ui-datepicker-hc-prev");
		var btnGoNextMonth = $("#" + thisDiv).find(".ui-datepicker-hc-next");
		var targetMonth = 0;
		var thisDate = new Date();
		var thisMonth = thisDate.getUTCMonth();
		var thisYear = thisDate.getUTCFullYear();
		btnGoTodayMonth.on("click", function(){
			if($("#" + thisDiv).attr("data-animation") == true) return;
			$.datepickerHC._gotoMonth(thisYear, thisMonth, true, inst);
		});
		btnGoPrevMonth.on("click", function(){
			//console.log("이전달")
			if($("#" + thisDiv).attr("data-animation") == true) return;
			var curYear = $("#" + thisDiv + " .ui-datepicker-hc-current").attr("data-current-year") * 1;
			var curMonth = $("#" + thisDiv + " .ui-datepicker-hc-current").attr("data-current-month") * 1 - 1;
			if(curMonth > 0){
				curMonth = curMonth - 1;
			}else{
				curMonth = 11;
				curYear = curYear - 1;
			}
			$.datepickerHC._gotoMonth(curYear, curMonth, true, inst);
		});
		btnGoNextMonth.on("click", function(){
			if($("#" + thisDiv).attr("data-animation") == true) return;
			var curYear = $("#" + thisDiv + " .ui-datepicker-hc-current").attr("data-current-year") * 1;
			var curMonth = $("#" + thisDiv + " .ui-datepicker-hc-current").attr("data-current-month") * 1 - 1;
			//console.log(curYear + " / " + curMonth)
			if(curYear == 0){
				curYear = $("#" + thisDiv).find(".ui-datepicker-hc-group").eq(0).find(".ui-datepicker-hc-title .ui-datepicker-hc-year").text() * 1;
				curMonth = $("#" + thisDiv).find(".ui-datepicker-hc-group").eq(0).find(".ui-datepicker-hc-title .ui-datepicker-hc-month").text() * 1 - 1;
			}else{
				if(curMonth < 11){
					curMonth = curMonth + 1;
				}else{
					curMonth = 0;
					curYear = curYear + 1;
				}
			}
			$.datepickerHC._gotoMonth(curYear, curMonth, true, inst);
		});
		/*
		if(inst.currentDay == 0){ //선택값 없음
			var goMonth = inst.selectedMonth;
			var goYear = inst.selectedYear;
			var goPosY = '';

			$("#" + thisDiv).find(".ui-datepicker-hc-group").each(function(index, element){
				var tgMonth = $(this).find("tbody tr").eq(1).find("td").attr("data-month");
				var tgYear = $(this).find("tbody tr").eq(1).find("td").eq(0).attr("data-year");
				if(tgMonth == goMonth && tgYear == goYear){
					goPosY = $("#" + thisDiv).find(".ui-datepicker-hc-group").eq(index).position().top;
				}
			});
			$("#" + thisDiv + " .ui-datepicker-hc-holder").scrollTop(goPosY);
		}else{ //선택값 있음
			var goMonth = inst.selectedMonth;
			var goYear = inst.selectedYear;
			var goPosY = '';
			$("#" + thisDiv).find(".ui-datepicker-hc-group").each(function(index, element){
				var tgMonth = $(this).find("tbody tr").eq(1).find("td").attr("data-month");
				var tgYear = $(this).find("tbody tr").eq(1).find("td").eq(0).attr("data-year");
				if(tgMonth == goMonth && tgYear == goYear){
					goPosY = $("#" + thisDiv).find(".ui-datepicker-hc-group").eq(index).position().top;
				}
			});
			$("#" + thisDiv + " .ui-datepicker-hc-holder").scrollTop(goPosY);
		}
		*/
	},

	// #6694 - don't focus the input if it's already focused
	// this breaks the change event in IE
	// Support: IE and jQuery <1.9
	_shouldFocusInput: function( inst ) {
		return inst.input && inst.input.is( ":visible" ) && !inst.input.is( ":disabled" ) && !inst.input.is( ":focus" );
	},

	/* Check positioning to remain on screen. */
	_checkOffset: function( inst, offset, isFixed ) {
		var dpWidth = inst.dpDiv.outerWidth(),
			dpHeight = inst.dpDiv.outerHeight(),
			inputWidth = inst.input ? inst.input.outerWidth() : 0,
			inputHeight = inst.input ? inst.input.outerHeight() : 0,
			viewWidth = document.documentElement.clientWidth + ( isFixed ? 0 : $( document ).scrollLeft() ),
			viewHeight = document.documentElement.clientHeight + ( isFixed ? 0 : $( document ).scrollTop() );

		offset.left -= ( this._get( inst, "isRTL" ) ? ( dpWidth - inputWidth ) : 0 );
		offset.left -= ( isFixed && offset.left === inst.input.offset().left ) ? $( document ).scrollLeft() : 0;
		offset.top -= ( isFixed && offset.top === ( inst.input.offset().top + inputHeight ) ) ? $( document ).scrollTop() : 0;

		// Now check if datepickerHC is showing outside window viewport - move to a better place if so.
		offset.left -= Math.min( offset.left, ( offset.left + dpWidth > viewWidth && viewWidth > dpWidth ) ?
			Math.abs( offset.left + dpWidth - viewWidth ) : 0 );
		offset.top -= Math.min( offset.top, ( offset.top + dpHeight > viewHeight && viewHeight > dpHeight ) ?
			Math.abs( dpHeight + inputHeight ) : 0 );

		return offset;
	},

	/* Find an object's position on the screen. */
	_findPos: function( obj ) {
		var position,
			inst = this._getInst( obj ),
			isRTL = this._get( inst, "isRTL" );

		while ( obj && ( obj.type === "hidden" || obj.nodeType !== 1 || $.expr.pseudos.hidden( obj ) ) ) {
			obj = obj[ isRTL ? "previousSibling" : "nextSibling" ];
		}

		position = $( obj ).offset();
		return [ position.left, position.top ];
	},

	/* Hide the date picker from view.
	 * @param  input  element - the input field attached to the date picker
	 */
	_hideDatepickerHC: function( input ) {
		var showAnim, duration, postProcess, onClose,
			inst = this._curInst;

		//console.log("_hideDatepickerHC")
		if ( !inst || ( input && inst !== $.data( input, "datepickerHC" ) ) ) {
			return;
		}

		if ( this._datepickerShowing ) {
			showAnim = this._get( inst, "showAnim" );
			duration = this._get( inst, "duration" );
			postProcess = function() {
				$.datepickerHC._tidyDialog( inst );
			};

			// DEPRECATED: after BC for 1.8.x $.effects[ showAnim ] is not needed
			if ( $.effects && ( $.effects.effect[ showAnim ] || $.effects[ showAnim ] ) ) {
				inst.dpDiv.hide( showAnim, $.datepickerHC._get( inst, "showOptions" ), duration, postProcess );
			} else {
				inst.dpDiv[ ( showAnim === "slideDown" ? "slideUp" :
					( showAnim === "fadeIn" ? "fadeOut" : "hide" ) ) ]( ( showAnim ? duration : null ), postProcess );
			}

			if ( !showAnim ) {
				postProcess();
			}
			this._datepickerShowing = false;

			onClose = this._get( inst, "onClose" );
			if ( onClose ) {
				onClose.apply( ( inst.input ? inst.input[ 0 ] : null ), [ ( inst.input ? inst.input.val() : "" ), inst ] );
			}

			this._lastInput = null;
			if ( this._inDialog ) {
				this._dialogInput.css( { position: "absolute", left: "0", top: "-100px" } );
				if ( $.blockUI ) {
					$.unblockUI();
					$( "body" ).append( this.dpDiv );
				}
			}
			this._inDialog = false;
		}

		inst.dpDiv.attr("data-scrolltop",""); //hc-ui : 추가
	},

	/* Tidy up after a dialog display. */
	_tidyDialog: function( inst ) {
		inst.dpDiv.removeClass( this._dialogClass ).off( ".ui-datepicker-hc-calendar" );
	},

	/* Close date picker if clicked elsewhere. */
	_checkExternalClick: function( event ) {
		if ( !$.datepickerHC._curInst ) {
			return;
		}
		var $target = $( event.target ),
			inst = $.datepickerHC._getInst( $target[ 0 ] );
		//console.log($.datepickerHC._curInst.settings.idRangeTo)
		//console.log($target[ 0 ].id)
		if ( ( ( $target[ 0 ].id !== $.datepickerHC._mainDivId &&
				//$target[ 0 ].id !== inst.settings.idRangeTo && 
				$target.parents( "#" + $.datepickerHC._mainDivId ).length === 0 &&
				!$target.hasClass( $.datepickerHC.markerClassName ) &&
				!$target.closest( "." + $.datepickerHC._triggerClass ).length &&
				//inst2.settings.idRangeTo !== $target[ 0 ].id  &&
				$.datepickerHC._datepickerShowing && !( $.datepickerHC._inDialog && $.blockUI ) ) ) ||
				//( $target.hasClass( $.datepickerHC.markerClassName ) && $.datepickerHC._curInst !== inst) ) { //hc-ui : 폼 바깐 클릭시 to 필드인지 체크
				( $target.hasClass( $.datepickerHC.markerClassName ) && $.datepickerHC._curInst !== inst && $.datepickerHC._curInst.settings.idRangeTo !== $target[ 0 ].id) ) {
					$.datepickerHC._hideDatepickerHC();
		}
	},

	/* Adjust one of the date sub-fields. */
	_adjustDate: function( id, offset, period ) {
		var target = $( id ),
			inst = this._getInst( target[ 0 ] );

		if ( this._isDisabledDatepickerHC( target[ 0 ] ) ) {
			return;
		}
		this._adjustInstDate( inst, offset, period );
		this._updateDatepickerHC( inst );
	},

	/* Action for current link. */
	_gotoToday: function( id ) {
		var date,
			target = $( id ),
			inst = this._getInst( target[ 0 ] );

		if ( this._get( inst, "gotoCurrent" ) && inst.currentDay ) {
			inst.selectedDay = inst.currentDay;
			inst.drawMonth = inst.selectedMonth = inst.currentMonth;
			inst.drawYear = inst.selectedYear = inst.currentYear;
		} else {
			date = new Date();
			inst.selectedDay = date.getDate();
			inst.drawMonth = inst.selectedMonth = date.getMonth();
			inst.drawYear = inst.selectedYear = date.getFullYear();
		}
		this._notifyChange( inst );
		this._adjustDate( target );
	},

	/* hc-ui :  달 단위 스크롤 이동 처리 */
	_gotoMonth: function(goYear, goMonth, motion, inst){
		var thisDiv = this._mainDivId;
		var goPosY = '';
		var speed = this._get( inst, "speedScroll" )
		//console.log("스피드: " + speed)
		//console.log("_gotoMonth : year/" + goYear + " -- month/" + goMonth);
		$("#" + thisDiv).find(".ui-datepicker-hc-group").each(function(index, element){
			var tgMonth = $(this).find(".ui-datepicker-hc-month").text() * 1 - 1;
			var tgYear = $(this).find(".ui-datepicker-hc-year").text();
			if(tgMonth == goMonth && tgYear == goYear){
				//console.log("타겟 달 index :" +  index)
				goPosY = $("#" + thisDiv).find(".ui-datepicker-hc-group").eq(index).position().top + 10 + $("#" + thisDiv + " .ui-datepicker-hc-holder").scrollTop();
				//console.log("타겟 달 posY : " + goPosY)
			}
		});
		//if(!inst.settings.rangeOneCal)
		if(motion){
			$("#" + thisDiv).attr("data-animation",true);
			$("#" + thisDiv + " .ui-datepicker-hc-holder").animate({scrollTop: goPosY}, speed, "easeOutExpo",function(){
				$("#" + thisDiv).attr("data-animation",false);
			})
			//$("#" + thisDiv + " .ui-datepicker-hc-holder").scrollTop(goPosY);
		}else{
			$("#" + thisDiv + " .ui-datepicker-hc-holder").scrollTop(goPosY);
		}
		
	},

	/* Action for selecting a new month/year. */
	_selectMonthYear: function( id, select, period ) {
		//console.log("_selectMonthYear")
		var target = $( id ),
			inst = this._getInst( target[ 0 ] );

		inst[ "selected" + ( period === "M" ? "Month" : "Year" ) ] =
		inst[ "draw" + ( period === "M" ? "Month" : "Year" ) ] =
			parseInt( select.options[ select.selectedIndex ].value, 10 );

		this._notifyChange( inst );
		this._adjustDate( target );
	},

	/* Action for selecting a day. */
	_selectDay: function( id, month, year, td ) {
		//console.log("_selectDay")
		var inst,
			target = $( id );

		if ( $( td ).hasClass( this._unselectableClass ) || this._isDisabledDatepickerHC( target[ 0 ] ) ) {
			return;
		}
		
		inst = this._getInst( target[ 0 ] );
		inst.selectedDay = inst.currentDay = parseInt( $( "a", td ).attr( "data-date" ) );
		inst.selectedMonth = inst.currentMonth = month;
		inst.selectedYear = inst.currentYear = year;
		this._selectDate( id, this._formatDate( inst,
			inst.currentDay, inst.currentMonth, inst.currentYear ), inst );
	},

	/* Erase the input field and hide the date picker. */
	_clearDate: function( id ) {
		var target = $( id );
		this._selectDate( target, "" );
	},

	/* hc-ui : 1캘린더 ui에서 첫번째 값 선택 후 두번째 값 선택 전 범위 색상 그리기 */
	_drawRangeHover: function(inst){
		var curY, curM, curD, //
		thisY, thisM, thisD, //
		rangeY, rangeM, rangeD; //
		//$("#" + inst.id).attr("data-selected-length") * 1
		//if(this._defaults.selectedDateLength == 1){ //1번째 값 선택되어 td hover 색상 그리기
		if($("#" + inst.id).attr("data-selected-length") * 1 == 1){ //1번째 값 선택되어 td hover 색상 그리기
				/* curY = inst.currentYear;
			curM = inst.currentMonth;
			curD = inst.currentDate; */
			inst.dpDiv.on("mouseover",".ui-datepicker-hc-calendar td.ui-datepicker-hc-hoverable", function(event){
				thisY = $(this).data("year");
				thisM = $(this).data("month");
				thisD = $(this).find("a").data("date");
				//console.log(typeof(thisY))
				//inst.dpDiv.find("")
				inst.dpDiv.find("td.ui-datepicker-hc-hoverable").removeClass("ui-datepicker-hc-range").removeClass("ui-datepicker-hc-range-end");
				inst.dpDiv.find("td.ui-datepicker-hc-hoverable").each(function(index, element){
					//console.log(index)
					//$(this).removeClass("ui-datepicker-hc-range-end")
					rangeY = $(this).data("year");
					rangeM = $(this).data("month");
					rangeD = $(this).find("a").data("date");
					if(rangeY <= thisY && rangeM <= thisM){
						if(rangeM == thisM){
							if(rangeD < thisD) $(this).addClass("ui-datepicker-hc-range")
						}else if(rangeM < thisM){
							$(this).addClass("ui-datepicker-hc-range")
						}
					}
				});
				$(this).removeClass("ui-datepicker-hc-range").addClass("ui-datepicker-hc-range-end");
				inst.dpDiv.find(".ui-datepicker-hc-current-day").addClass("ui-datepicker-hc-range-start");
			});
		//}else if(this._defaults.selectedDateLength == 2){ //2번째 값까지 선택되어 선택값 구간 td 색상 그리기
		}else if($("#" + inst.id).attr("data-selected-length") * 1 == 2){ //2번째 값까지 선택되어 선택값 구간 td 색상 그리기

		}
		/*
		console.log(inst);
		curY = inst.currentYear;
		curM = inst.currentMonth;
		curD = inst.currentDate;
		inst.dpDiv.on("mouseover",".ui-datepicker-hc-calendar td.ui-datepicker-hc-hoverable", function(event){
			var thisY = $(this).data("year");
			var thisM = $(this).data("month");
			var thisD = $(this).find("a").data("date");
			console.log(thisM)
		});
		*/
	},

	/* hc-ui : 1캘린더 ui에서 범위 색상 지우기 */
	_removeRangeHover: function(){

	},

	/* Update the input field with the selected date. */
	_selectDate: function( id, dateStr, inst ) {
		//console.log("선택됨 : " + id)
		//console.log("선택됨 : " + dateStr)
		var onSelect,
			target = $( id ),
			inst = this._getInst( target[ 0 ] );
		
		dateStr = ( dateStr != null ? dateStr : this._formatDate( inst ) );
		/* hc-ui : 기존 if문
		if ( inst.input ) {
				inst.input.val( dateStr );
		}
		*/
		//hc-ui : rangeOneCal일 경우 두번째 선택 시 to 필드에 값 넣어줌.
		if ( inst.input ) {
			if(!inst.settings.rangeOneCal){
				inst.input.val( dateStr );
			}else{
				//hc-ui : $("#" + inst.settings.idRangeTo).val( dateStr );

				//this._defaults.selectedDateLength = this._defaults.selectedDateLength + 1;
				var lengthNum = $(id).attr("data-selected-length") * 1;
				$(id).attr("data-selected-length", lengthNum + 1);
				//if(this._defaults.selectedDateLength == 3){ //2번째 값까지 선택된 상태
				if($(id).attr("data-selected-length") * 1 == 3){ //2번째 값까지 선택된 상태
					//this._defaults.selectedDateLength = 1;
					$(id).attr("data-selected-length", 1);
					inst.input.val( dateStr );
					$("#" + inst.settings.idRangeTo).val("");
					$("#" + inst.settings.idRangeTo).attr("data-selected-year", "");
					$("#" + inst.settings.idRangeTo).attr("data-selected-month", "");
					$("#" + inst.settings.idRangeTo).attr("data-selected-day", "");
					//console.log(this._defaults.selectedDateLength)
				//}else if(this._defaults.selectedDateLength == 2){ //2번째 값 선택시
				}else if($(id).attr("data-selected-length") * 1 == 2){ //2번째 값 선택시
					//console.log("2번째 값 선택");
					//console.log(inst)
					$("#" + inst.settings.idRangeTo).val( dateStr );
					$("#" + inst.settings.idRangeTo).attr("data-selected-year", inst.selectedYear)
					$("#" + inst.settings.idRangeTo).attr("data-selected-month", inst.selectedMonth)
					$("#" + inst.settings.idRangeTo).attr("data-selected-day", inst.selectedDay)
					$("#" + inst.settings.idRangeTo).trigger( "change" )
				//}else if(this._defaults.selectedDateLength == 1){ //1번째 값 선택시
				}else if($(id).attr("data-selected-length") * 1 == 1){ //1번째 값 선택시
					//console.log("값 선택 갯수 :" + this._defaults.selectedDateLength);
					inst.input.val( dateStr );
					//console.log(inst)
					/*
					console.log(inst);
					curY = inst.currentYear;
					curM = inst.currentMonth;
					curD = inst.currentDate;
					inst.dpDiv.on("mouseover",".ui-datepicker-hc-calendar td.ui-datepicker-hc-hoverable", function(event){
						var thisY = $(this).data("year");
						var thisM = $(this).data("month");
						var thisD = $(this).find("a").data("date");
						console.log(thisM)
					});
					*/
				}
			}
		}
		this._updateAlternate( inst );
		
		onSelect = this._get( inst, "onSelect" );
		//inst.input.trigger( "change" ); 
		if ( onSelect ) {
			//console.log("_selectDate - onSelect")
			onSelect.apply( ( inst.input ? inst.input[ 0 ] : null ), [ dateStr, inst ] );  // trigger custom callback
			inst.input.trigger( "change" )
		} else if ( inst.input ) {
			inst.input.trigger( "change" ); // fire the change event
		}
		
		if ( inst.inline ) {
			this._updateDatepickerHC( inst );
		} else {
			if(!inst.settings.rangeOneCal){ //hc-ui: rangeOneCal 값  false
				this._hideDatepickerHC(); 
				this._lastInput = inst.input[ 0 ];
				if ( typeof( inst.input[ 0 ] ) !== "object" ) {
					inst.input.trigger( "focus" ); // restore focus
				}
				this._lastInput = null;
			}else{ //hc-ui: rangeOneCal 값  true
				//if(this._defaults.selectedDateLength == 2){ //2번째 값 선택시
				if($(id).attr("data-selected-length") * 1 == 2){ //2번째 값 선택시
					this._hideDatepickerHC();
					this._hideDatepickerHC();
					this._updateDatepickerHC( inst );
					//this._defaults.selectedDateLength = 0;
					
				}else{ //1번째 값 선택시
					//console.log("값 선택 갯수 :" + this._defaults.selectedDateLength);
					this._updateDatepickerHC( inst );
				}
				//this._updateDatepickerHC( inst );
			}
		}
	},

	/* Update any alternate field to synchronise with the main field. */
	_updateAlternate: function( inst ) {
		var altFormat, date, dateStr,
			altField = this._get( inst, "altField" );

		if ( altField ) { // update alternate field too
			altFormat = this._get( inst, "altFormat" ) || this._get( inst, "dateFormat" );
			date = this._getDate( inst );
			dateStr = this.formatDate( altFormat, date, this._getFormatConfig( inst ) );
			$( document ).find( altField ).val( dateStr );
		}
	},

	/* Set as beforeShowDay function to prevent selection of weekends.
	 * @param  date  Date - the date to customise
	 * @return [boolean, string] - is this date selectable?, what is its CSS class?
	 */
	noWeekends: function( date ) {
		var day = date.getDay();
		return [ ( day > 0 && day < 6 ), "" ];
	},

	/* Set as calculateWeek to determine the week of the year based on the ISO 8601 definition.
	 * @param  date  Date - the date to get the week for
	 * @return  number - the number of the week within the year that contains this date
	 */
	iso8601Week: function( date ) {
		var time,
			checkDate = new Date( date.getTime() );

		// Find Thursday of this week starting on Monday
		checkDate.setDate( checkDate.getDate() + 4 - ( checkDate.getDay() || 7 ) );

		time = checkDate.getTime();
		checkDate.setMonth( 0 ); // Compare with Jan 1
		checkDate.setDate( 1 );
		return Math.floor( Math.round( ( time - checkDate ) / 86400000 ) / 7 ) + 1;
	},

	/* Parse a string value into a date object.
	 * See formatDate below for the possible formats.
	 *
	 * @param  format string - the expected format of the date
	 * @param  value string - the date in the above format
	 * @param  settings Object - attributes include:
	 *					shortYearCutoff  number - the cutoff year for determining the century (optional)
	 *					dayNamesShort	string[7] - abbreviated names of the days from Sunday (optional)
	 *					dayNames		string[7] - names of the days from Sunday (optional)
	 *					monthNamesShort string[12] - abbreviated names of the months (optional)
	 *					monthNames		string[12] - names of the months (optional)
	 * @return  Date - the extracted date value or null if value is blank
	 */
	parseDate: function( format, value, settings ) {
		if ( format == null || value == null ) {
			throw "Invalid arguments";
		}

		value = ( typeof value === "object" ? value.toString() : value + "" );
		if ( value === "" ) {
			return null;
		}

		var iFormat, dim, extra,
			iValue = 0,
			shortYearCutoffTemp = ( settings ? settings.shortYearCutoff : null ) || this._defaults.shortYearCutoff,
			shortYearCutoff = ( typeof shortYearCutoffTemp !== "string" ? shortYearCutoffTemp :
				new Date().getFullYear() % 100 + parseInt( shortYearCutoffTemp, 10 ) ),
			dayNamesShort = ( settings ? settings.dayNamesShort : null ) || this._defaults.dayNamesShort,
			dayNames = ( settings ? settings.dayNames : null ) || this._defaults.dayNames,
			monthNamesShort = ( settings ? settings.monthNamesShort : null ) || this._defaults.monthNamesShort,
			monthNames = ( settings ? settings.monthNames : null ) || this._defaults.monthNames,
			year = -1,
			month = -1,
			day = -1,
			doy = -1,
			literal = false,
			date,

			// Check whether a format character is doubled
			lookAhead = function( match ) {
				var matches = ( iFormat + 1 < format.length && format.charAt( iFormat + 1 ) === match );
				if ( matches ) {
					iFormat++;
				}
				return matches;
			},

			// Extract a number from the string value
			getNumber = function( match ) {
				var isDoubled = lookAhead( match ),
					size = ( match === "@" ? 14 : ( match === "!" ? 20 :
					( match === "y" && isDoubled ? 4 : ( match === "o" ? 3 : 2 ) ) ) ),
					minSize = ( match === "y" ? size : 1 ),
					digits = new RegExp( "^\\d{" + minSize + "," + size + "}" ),
					num = value.substring( iValue ).match( digits );
				if ( !num ) {
					throw "Missing number at position " + iValue;
				}
				iValue += num[ 0 ].length;
				return parseInt( num[ 0 ], 10 );
			},

			// Extract a name from the string value and convert to an index
			getName = function( match, shortNames, longNames ) {
				var index = -1,
					names = $.map( lookAhead( match ) ? longNames : shortNames, function( v, k ) {
						return [ [ k, v ] ];
					} ).sort( function( a, b ) {
						return -( a[ 1 ].length - b[ 1 ].length );
					} );

				$.each( names, function( i, pair ) {
					var name = pair[ 1 ];
					if ( value.substr( iValue, name.length ).toLowerCase() === name.toLowerCase() ) {
						index = pair[ 0 ];
						iValue += name.length;
						return false;
					}
				} );
				if ( index !== -1 ) {
					return index + 1;
				} else {
					throw "Unknown name at position " + iValue;
				}
			},

			// Confirm that a literal character matches the string value
			checkLiteral = function() {
				if ( value.charAt( iValue ) !== format.charAt( iFormat ) ) {
					throw "Unexpected literal at position " + iValue;
				}
				iValue++;
			};

		for ( iFormat = 0; iFormat < format.length; iFormat++ ) {
			if ( literal ) {
				if ( format.charAt( iFormat ) === "'" && !lookAhead( "'" ) ) {
					literal = false;
				} else {
					checkLiteral();
				}
			} else {
				switch ( format.charAt( iFormat ) ) {
					case "d":
						day = getNumber( "d" );
						break;
					case "D":
						getName( "D", dayNamesShort, dayNames );
						break;
					case "o":
						doy = getNumber( "o" );
						break;
					case "m":
						month = getNumber( "m" );
						break;
					case "M":
						month = getName( "M", monthNamesShort, monthNames );
						break;
					case "y":
						year = getNumber( "y" );
						break;
					case "@":
						date = new Date( getNumber( "@" ) );
						year = date.getFullYear();
						month = date.getMonth() + 1;
						day = date.getDate();
						break;
					case "!":
						date = new Date( ( getNumber( "!" ) - this._ticksTo1970 ) / 10000 );
						year = date.getFullYear();
						month = date.getMonth() + 1;
						day = date.getDate();
						break;
					case "'":
						if ( lookAhead( "'" ) ) {
							checkLiteral();
						} else {
							literal = true;
						}
						break;
					default:
						checkLiteral();
				}
			}
		}

		if ( iValue < value.length ) {
			extra = value.substr( iValue );
			if ( !/^\s+/.test( extra ) ) {
				throw "Extra/unparsed characters found in date: " + extra;
			}
		}

		if ( year === -1 ) {
			year = new Date().getFullYear();
		} else if ( year < 100 ) {
			year += new Date().getFullYear() - new Date().getFullYear() % 100 +
				( year <= shortYearCutoff ? 0 : -100 );
		}

		if ( doy > -1 ) {
			month = 1;
			day = doy;
			do {
				dim = this._getDaysInMonth( year, month - 1 );
				if ( day <= dim ) {
					break;
				}
				month++;
				day -= dim;
			} while ( true );
		}

		date = this._daylightSavingAdjust( new Date( year, month - 1, day ) );
		if ( date.getFullYear() !== year || date.getMonth() + 1 !== month || date.getDate() !== day ) {
			throw "Invalid date"; // E.g. 31/02/00
		}
		return date;
	},

	/* Standard date formats. */
	ATOM: "yy-mm-dd", // RFC 3339 (ISO 8601)
	COOKIE: "D, dd M yy",
	ISO_8601: "yy-mm-dd",
	RFC_822: "D, d M y",
	RFC_850: "DD, dd-M-y",
	RFC_1036: "D, d M y",
	RFC_1123: "D, d M yy",
	RFC_2822: "D, d M yy",
	RSS: "D, d M y", // RFC 822
	TICKS: "!",
	TIMESTAMP: "@",
	W3C: "yy-mm-dd", // ISO 8601

	_ticksTo1970: ( ( ( 1970 - 1 ) * 365 + Math.floor( 1970 / 4 ) - Math.floor( 1970 / 100 ) +
		Math.floor( 1970 / 400 ) ) * 24 * 60 * 60 * 10000000 ),

	/* Format a date object into a string value.
	 * The format can be combinations of the following:
	 * d  - day of month (no leading zero)
	 * dd - day of month (two digit)
	 * o  - day of year (no leading zeros)
	 * oo - day of year (three digit)
	 * D  - day name short
	 * DD - day name long
	 * m  - month of year (no leading zero)
	 * mm - month of year (two digit)
	 * M  - month name short
	 * MM - month name long
	 * y  - year (two digit)
	 * yy - year (four digit)
	 * @ - Unix timestamp (ms since 01/01/1970)
	 * ! - Windows ticks (100ns since 01/01/0001)
	 * "..." - literal text
	 * '' - single quote
	 *
	 * @param  format string - the desired format of the date
	 * @param  date Date - the date value to format
	 * @param  settings Object - attributes include:
	 *					dayNamesShort	string[7] - abbreviated names of the days from Sunday (optional)
	 *					dayNames		string[7] - names of the days from Sunday (optional)
	 *					monthNamesShort string[12] - abbreviated names of the months (optional)
	 *					monthNames		string[12] - names of the months (optional)
	 * @return  string - the date in the above format
	 */
	formatDate: function( format, date, settings ) {
		if ( !date ) {
			return "";
		}

		var iFormat,
			dayNamesShort = ( settings ? settings.dayNamesShort : null ) || this._defaults.dayNamesShort,
			dayNames = ( settings ? settings.dayNames : null ) || this._defaults.dayNames,
			monthNamesShort = ( settings ? settings.monthNamesShort : null ) || this._defaults.monthNamesShort,
			monthNames = ( settings ? settings.monthNames : null ) || this._defaults.monthNames,

			// Check whether a format character is doubled
			lookAhead = function( match ) {
				var matches = ( iFormat + 1 < format.length && format.charAt( iFormat + 1 ) === match );
				if ( matches ) {
					iFormat++;
				}
				return matches;
			},

			// Format a number, with leading zero if necessary
			formatNumber = function( match, value, len ) {
				var num = "" + value;
				if ( lookAhead( match ) ) {
					while ( num.length < len ) {
						num = "0" + num;
					}
				}
				return num;
			},

			// Format a name, short or long as requested
			formatName = function( match, value, shortNames, longNames ) {
				return ( lookAhead( match ) ? longNames[ value ] : shortNames[ value ] );
			},
			output = "",
			literal = false;

		if ( date ) {
			for ( iFormat = 0; iFormat < format.length; iFormat++ ) {
				if ( literal ) {
					if ( format.charAt( iFormat ) === "'" && !lookAhead( "'" ) ) {
						literal = false;
					} else {
						output += format.charAt( iFormat );
					}
				} else {
					switch ( format.charAt( iFormat ) ) {
						case "d":
							output += formatNumber( "d", date.getDate(), 2 );
							break;
						case "D":
							output += formatName( "D", date.getDay(), dayNamesShort, dayNames );
							break;
						case "o":
							output += formatNumber( "o",
								Math.round( ( new Date( date.getFullYear(), date.getMonth(), date.getDate() ).getTime() - new Date( date.getFullYear(), 0, 0 ).getTime() ) / 86400000 ), 3 );
							break;
						case "m":
							output += formatNumber( "m", date.getMonth() + 1, 2 );
							break;
						case "M":
							output += formatName( "M", date.getMonth(), monthNamesShort, monthNames );
							break;
						case "y":
							output += ( lookAhead( "y" ) ? date.getFullYear() :
								( date.getFullYear() % 100 < 10 ? "0" : "" ) + date.getFullYear() % 100 );
							break;
						case "@":
							output += date.getTime();
							break;
						case "!":
							output += date.getTime() * 10000 + this._ticksTo1970;
							break;
						case "'":
							if ( lookAhead( "'" ) ) {
								output += "'";
							} else {
								literal = true;
							}
							break;
						default:
							output += format.charAt( iFormat );
					}
				}
			}
		}
		return output;
	},

	/* Extract all possible characters from the date format. */
	_possibleChars: function( format ) {
		var iFormat,
			chars = "",
			literal = false,

			// Check whether a format character is doubled
			lookAhead = function( match ) {
				var matches = ( iFormat + 1 < format.length && format.charAt( iFormat + 1 ) === match );
				if ( matches ) {
					iFormat++;
				}
				return matches;
			};

		for ( iFormat = 0; iFormat < format.length; iFormat++ ) {
			if ( literal ) {
				if ( format.charAt( iFormat ) === "'" && !lookAhead( "'" ) ) {
					literal = false;
				} else {
					chars += format.charAt( iFormat );
				}
			} else {
				switch ( format.charAt( iFormat ) ) {
					case "d": case "m": case "y": case "@":
						chars += "0123456789";
						break;
					case "D": case "M":
						return null; // Accept anything
					case "'":
						if ( lookAhead( "'" ) ) {
							chars += "'";
						} else {
							literal = true;
						}
						break;
					default:
						chars += format.charAt( iFormat );
				}
			}
		}
		return chars;
	},

	/* Get a setting value, defaulting if necessary. */
	_get: function( inst, name ) {
		return inst.settings[ name ] !== undefined ?
			inst.settings[ name ] : this._defaults[ name ];
	},

	/* Parse existing date and initialise date picker. */
	_setDateFromField: function( inst, noDefault ) {
		if ( inst.input.val() === inst.lastVal ) {
			return;
		}

		var dateFormat = this._get( inst, "dateFormat" ),
			dates = inst.lastVal = inst.input ? inst.input.val() : null,
			defaultDate = this._getDefaultDate( inst ),
			date = defaultDate,
			settings = this._getFormatConfig( inst );

		try {
			date = this.parseDate( dateFormat, dates, settings ) || defaultDate;
		} catch ( event ) {
			dates = ( noDefault ? "" : dates );
		}
		inst.selectedDay = date.getDate();
		inst.drawMonth = inst.selectedMonth = date.getMonth();
		inst.drawYear = inst.selectedYear = date.getFullYear();
		inst.currentDay = ( dates ? date.getDate() : 0 );
		inst.currentMonth = ( dates ? date.getMonth() : 0 );
		inst.currentYear = ( dates ? date.getFullYear() : 0 );
		this._adjustInstDate( inst );
	},

	/* Retrieve the default date shown on opening. */
	_getDefaultDate: function( inst ) {
		return this._restrictMinMax( inst,
			this._determineDate( inst, this._get( inst, "defaultDate" ), new Date() ) );
	},

	/* A date may be specified as an exact value or a relative one. */
	_determineDate: function( inst, date, defaultDate ) {
		var offsetNumeric = function( offset ) {
				var date = new Date();
				date.setDate( date.getDate() + offset );
				return date;
			},
			offsetString = function( offset ) {
				try {
					return $.datepickerHC.parseDate( $.datepickerHC._get( inst, "dateFormat" ),
						offset, $.datepickerHC._getFormatConfig( inst ) );
				} catch ( e ) {

					// Ignore
				}

				var date = ( offset.toLowerCase().match( /^c/ ) ?
					$.datepickerHC._getDate( inst ) : null ) || new Date(),
					year = date.getFullYear(),
					month = date.getMonth(),
					day = date.getDate(),
					pattern = /([+\-]?[0-9]+)\s*(d|D|w|W|m|M|y|Y)?/g,
					matches = pattern.exec( offset );

				while ( matches ) {
					switch ( matches[ 2 ] || "d" ) {
						case "d" : case "D" :
							day += parseInt( matches[ 1 ], 10 ); break;
						case "w" : case "W" :
							day += parseInt( matches[ 1 ], 10 ) * 7; break;
						case "m" : case "M" :
							month += parseInt( matches[ 1 ], 10 );
							day = Math.min( day, $.datepickerHC._getDaysInMonth( year, month ) );
							break;
						case "y": case "Y" :
							year += parseInt( matches[ 1 ], 10 );
							day = Math.min( day, $.datepickerHC._getDaysInMonth( year, month ) );
							break;
					}
					matches = pattern.exec( offset );
				}
				return new Date( year, month, day );
			},
			newDate = ( date == null || date === "" ? defaultDate : ( typeof date === "string" ? offsetString( date ) :
				( typeof date === "number" ? ( isNaN( date ) ? defaultDate : offsetNumeric( date ) ) : new Date( date.getTime() ) ) ) );

		newDate = ( newDate && newDate.toString() === "Invalid Date" ? defaultDate : newDate );
		if ( newDate ) {
			newDate.setHours( 0 );
			newDate.setMinutes( 0 );
			newDate.setSeconds( 0 );
			newDate.setMilliseconds( 0 );
		}
		return this._daylightSavingAdjust( newDate );
	},

	/* Handle switch to/from daylight saving.
	 * Hours may be non-zero on daylight saving cut-over:
	 * > 12 when midnight changeover, but then cannot generate
	 * midnight datetime, so jump to 1AM, otherwise reset.
	 * @param  date  (Date) the date to check
	 * @return  (Date) the corrected date
	 */
	_daylightSavingAdjust: function( date ) {
		if ( !date ) {
			return null;
		}
		date.setHours( date.getHours() > 12 ? date.getHours() + 2 : 0 );
		return date;
	},

	/* Set the date(s) directly. */
	_setDate: function( inst, date, noChange ) {
		var clear = !date,
			origMonth = inst.selectedMonth,
			origYear = inst.selectedYear,
			newDate = this._restrictMinMax( inst, this._determineDate( inst, date, new Date() ) );

		inst.selectedDay = inst.currentDay = newDate.getDate();
		inst.drawMonth = inst.selectedMonth = inst.currentMonth = newDate.getMonth();
		inst.drawYear = inst.selectedYear = inst.currentYear = newDate.getFullYear();
		if ( ( origMonth !== inst.selectedMonth || origYear !== inst.selectedYear ) && !noChange ) {
			this._notifyChange( inst );
		}
		this._adjustInstDate( inst );
		if ( inst.input ) {
			inst.input.val( clear ? "" : this._formatDate( inst ) );
		}
	},

	/* Retrieve the date(s) directly. */
	_getDate: function( inst ) {
		var startDate = ( !inst.currentYear || ( inst.input && inst.input.val() === "" ) ? null :
			this._daylightSavingAdjust( new Date(
			inst.currentYear, inst.currentMonth, inst.currentDay ) ) );
			return startDate;
	},

	/* Attach the onxxx handlers.  These are declared statically so
	 * they work with static code transformers like Caja.
	 */
	_attachHandlers: function( inst ) {
		var stepMonths = this._get( inst, "stepMonths" ),
			id = "#" + inst.id.replace( /\\\\/g, "\\" );
		inst.dpDiv.find( "[data-handler]" ).map( function() {
			var handler = {
				prev: function() {
					$.datepickerHC._adjustDate( id, -stepMonths, "M" );
				},
				next: function() {
					$.datepickerHC._adjustDate( id, +stepMonths, "M" );
				},
				hide: function() {
					$.datepickerHC._hideDatepickerHC();
				},
				today: function() {
					$.datepickerHC._gotoToday( id );
				},
				selectDay: function() {
					$.datepickerHC._selectDay( id, +this.getAttribute( "data-month" ), +this.getAttribute( "data-year" ), this );
					return false;
				},
				selectMonth: function() {
					$.datepickerHC._selectMonthYear( id, this, "M" );
					return false;
				},
				selectYear: function() {
					$.datepickerHC._selectMonthYear( id, this, "Y" );
					return false;
				}
			};
			$( this ).on( this.getAttribute( "data-event" ), handler[ this.getAttribute( "data-handler" ) ] );
		} );
	},

	/* Generate the HTML for the current state of the date picker. */
	_generateHTML: function( inst ) {
		//console.log("그리기")
		var maxDraw, prevText, prev, nextText, next, currentText, gotoDate,
			controls, buttonPanel, firstDay, showWeek, dayNames, dayNamesMin,
			monthNames, monthNamesShort, beforeShowDay, showOtherMonths,
			selectOtherMonths, defaultDate, html, dow, row, group, col, selectedDate,
			cornerClass, calender, thead, day, daysInMonth, leadDays, curRows, numRows,
			printDate, dRow, tbody, daySettings, otherMonth, unselectable,
			minSelectableDate,  // hc-ui  minSelectableDate 추가
			maxSelectableDate,  // hc-ui  maxSelectableDate 추가
			tdHoverable, //hc-ui tdHoverable 추가. 1캘린더 range 설정 시 from 값 선택 시 이후 값 hover 시각적 처리 위한 클래스 추가
			tempDate = new Date(),
			today = this._daylightSavingAdjust(
				new Date( tempDate.getFullYear(), tempDate.getMonth(), tempDate.getDate() ) ), // clear time
			isRTL = this._get( inst, "isRTL" ),
			showButtonPanel = this._get( inst, "showButtonPanel" ),
			hideIfNoPrevNext = this._get( inst, "hideIfNoPrevNext" ),
			navigationAsDateFormat = this._get( inst, "navigationAsDateFormat" ),
			numMonths = this._getNumberOfMonths( inst ),
			showCurrentAtPos = this._get( inst, "showCurrentAtPos" ),
			stepMonths = this._get( inst, "stepMonths" ),
			isMultiMonth = ( numMonths[ 0 ] !== 1 || numMonths[ 1 ] !== 1 ),
			currentDate = this._daylightSavingAdjust( ( !inst.currentDay ? new Date( 9999, 9, 9 ) :
				new Date( inst.currentYear, inst.currentMonth, inst.currentDay ) ) ),
			currentYearTo = null, // hc-ui 추가 : 1캘린더로 range 설정 시 선택된 to 값 중 y
			currentMonthTo = null, // hc-ui 추가 : 1캘린더로 range 설정 시 선택된 to 값 중 m
			currentDayTo = null, // hc-ui 추가 : 1캘린더로 range 설정 시 선택된 to 값 중 d
			currentDateTo = null , // hc-ui 추가  : 1캘린더로 range 설정 시 선택된 to 값
			minDate = this._getMinMaxDate( inst, "min" ),
			maxDate = this._getMinMaxDate( inst, "max" ),
			drawMonth = inst.drawMonth - showCurrentAtPos,
			drawYear = inst.drawYear;
			//console.log(currentDateTo)
		//console.log("_generateHTML currentDate : " + currentDate);
		//$(id).attr("data-selected-length")
		//if(inst.settings.rangeOneCal && this._defaults.selectedDateLength == 2){ // hc-ui : 1캘린더로 range 설정시 선택값이 2개이면 두번째 선택값 달력에 표현하기 위해 currentDayTo 값 정의
		if(inst.settings.rangeOneCal && $("#" + inst.id).attr("data-selected-length") * 1 == 2){ // hc-ui : 1캘린더로 range 설정시 선택값이 2개이면 두번째 선택값 달력에 표현하기 위해 currentDayTo 값 정의
				//console.log(inst.settings.idRangeTo)
			/* currentYearTo = $("#" + inst.settings.idRangeTo).data("selectedYear");
			currentMonthTo = $("#" + inst.settings.idRangeTo).data("selectedMonth");
			currentDayTo = $("#" + inst.settings.idRangeTo).data("selectedDay"); */
			currentYearTo = $("#" + inst.settings.idRangeTo).attr("data-selected-year") * 1;
			currentMonthTo = $("#" + inst.settings.idRangeTo).attr("data-selected-month") * 1;
			currentDayTo = $("#" + inst.settings.idRangeTo).attr("data-selected-day") * 1;
			// console.log("gen html 2개값 선택시")
			// console.log(currentYearTo)
			// console.log($("#" + inst.settings.idRangeTo).data("selectedYear"))
			// console.log(currentMonthTo)
			// console.log($("#" + inst.settings.idRangeTo).data("selectedMonth"))
			// console.log(currentDayTo)
			//console.log($("#" + inst.settings.idRangeTo).data("selectedDay"))
			currentDateTo = this._daylightSavingAdjust( ( !currentDayTo ? new Date( 9999, 9, 9 ) :
				new Date( currentYearTo, currentMonthTo, currentDayTo ) ) );
			//console.log(currentDateTo)
		}
		//var virtualToDate = $("#rcpd4").val()
		if ( drawMonth < 0 ) {
			drawMonth += 12;
			drawYear--;
		}
		if ( maxDate ) {
			maxDraw = this._daylightSavingAdjust( new Date( maxDate.getFullYear(),
				maxDate.getMonth() - ( numMonths[ 0 ] * numMonths[ 1 ] ) + 1, maxDate.getDate() ) );
			maxDraw = ( minDate && maxDraw < minDate ? minDate : maxDraw );
			while ( this._daylightSavingAdjust( new Date( drawYear, drawMonth, 1 ) ) > maxDraw ) {
				drawMonth--;
				if ( drawMonth < 0 ) {
					drawMonth = 11;
					drawYear--;
				}
			}
		}
		inst.drawMonth = drawMonth;
		inst.drawYear = drawYear;

		prevText = this._get( inst, "prevText" );
		prevText = ( !navigationAsDateFormat ? prevText : this.formatDate( prevText,
			this._daylightSavingAdjust( new Date( drawYear, drawMonth - stepMonths, 1 ) ),
			this._getFormatConfig( inst ) ) );

		if ( this._canAdjustMonth( inst, -1, drawYear, drawMonth ) ) {
			prev = $( "<a>" )
				.attr( {
					"class": "ui-datepicker-hc-prev ui-corner-all",
					"data-handler": "prev",
					"data-event": "click",
					title: prevText
				} )
				.append(
					$( "<span>" )
						.addClass( "ui-icon ui-icon-circle-triangle-" +
							( isRTL ? "e" : "w" ) )
						.text( prevText )
				)[ 0 ].outerHTML;
		} else if ( hideIfNoPrevNext ) {
			prev = "";
		} else {
			prev = $( "<a>" )
				.attr( {
					"class": "ui-datepicker-hc-prev ui-corner-all ui-state-disabled",
					title: prevText
				} )
				.append(
					$( "<span>" )
						.addClass( "ui-icon ui-icon-circle-triangle-" +
							( isRTL ? "e" : "w" ) )
						.text( prevText )
				)[ 0 ].outerHTML;
		}

		nextText = this._get( inst, "nextText" );
		nextText = ( !navigationAsDateFormat ? nextText : this.formatDate( nextText,
			this._daylightSavingAdjust( new Date( drawYear, drawMonth + stepMonths, 1 ) ),
			this._getFormatConfig( inst ) ) );

		if ( this._canAdjustMonth( inst, +1, drawYear, drawMonth ) ) {
			next = $( "<a>" )
				.attr( {
					"class": "ui-datepicker-hc-next ui-corner-all",
					"data-handler": "next",
					"data-event": "click",
					title: nextText
				} )
				.append(
					$( "<span>" )
						.addClass( "ui-icon ui-icon-circle-triangle-" +
							( isRTL ? "w" : "e" ) )
						.text( nextText )
				)[ 0 ].outerHTML;
		} else if ( hideIfNoPrevNext ) {
			next = "";
		} else {
			next = $( "<a>" )
				.attr( {
					"class": "ui-datepicker-hc-next ui-corner-all ui-state-disabled",
					title: nextText
				} )
				.append(
					$( "<span>" )
						.attr( "class", "ui-icon ui-icon-circle-triangle-" +
							( isRTL ? "w" : "e" ) )
						.text( nextText )
				)[ 0 ].outerHTML;
		}

		currentText = this._get( inst, "currentText" );
		gotoDate = ( this._get( inst, "gotoCurrent" ) && inst.currentDay ? currentDate : today );
		currentText = ( !navigationAsDateFormat ? currentText :
			this.formatDate( currentText, gotoDate, this._getFormatConfig( inst ) ) );

		controls = "";
		if ( !inst.inline ) {
			controls = $( "<button>" )
				.attr( {
					type: "button",
					"class": "ui-datepicker-hc-close ui-state-default ui-priority-primary ui-corner-all",
					"data-handler": "hide",
					"data-event": "click"
				} )
				.text( this._get( inst, "closeText" ) )[ 0 ].outerHTML;
		}

		buttonPanel = "";
		if ( showButtonPanel ) {
			buttonPanel = $( "<div class='ui-datepicker-hc-buttonpane ui-widget-content'>" )
				.append( isRTL ? controls : "" )
				.append( this._isInRange( inst, gotoDate ) ?
					$( "<button>" )
						.attr( {
							type: "button",
							"class": "ui-datepicker-hc-current ui-state-default ui-priority-secondary ui-corner-all",
							"data-handler": "today",
							"data-event": "click"
						} )
						.text( currentText ) :
					"" )
				.append( isRTL ? "" : controls )[ 0 ].outerHTML;
		}

		firstDay = parseInt( this._get( inst, "firstDay" ), 10 );
		firstDay = ( isNaN( firstDay ) ? 0 : firstDay );

		minSelectableDate = this._get( inst, "minSelectableDate" );
		maxSelectableDate = this._get( inst, "maxSelectableDate" );
		showWeek = this._get( inst, "showWeek" );
		dayNames = this._get( inst, "dayNames" );
		dayNamesMin = this._get( inst, "dayNamesMin" );
		monthNames = this._get( inst, "monthNames" );
		monthNamesShort = this._get( inst, "monthNamesShort" );
		beforeShowDay = this._get( inst, "beforeShowDay" );
		showOtherMonths = this._get( inst, "showOtherMonths" );
		selectOtherMonths = this._get( inst, "selectOtherMonths" );
		defaultDate = this._getDefaultDate( inst );
		html = "";
		//hc-ui:스크롤 처리 위해 엘리먼트 3겹추가
		html += "<div class='ui-datepicker-hc-container'>";
		html += "<div class='ui-datepicker-hc-holder' style='margin-right:-" + commonUi.layerPop.getScrollbarWidth() + "px'>";
		//html += "<div class='ui-datepicker-hc-holder' style='margin-right:0px'>";
		html += "<div class='ui-datepicker-hc-scroll'>";

		for ( row = 0; row < numMonths[ 0 ]; row++ ) {
			group = "";
			this.maxRows = 4;
			for ( col = 0; col < numMonths[ 1 ]; col++ ) {

				leadDays = ( this._getFirstDayOfMonth( drawYear, drawMonth ) - firstDay + 7 ) % 7;
				printDate = this._daylightSavingAdjust( new Date( drawYear, drawMonth, 1 - leadDays ) );

				selectedDate = this._daylightSavingAdjust( new Date( drawYear, drawMonth, inst.selectedDay ) );
				cornerClass = " ui-corner-all";
				calender = "";
				if ( isMultiMonth ) {
					calender += "<div class='ui-datepicker-hc-group";
					if ( numMonths[ 1 ] > 1 ) {
						switch ( col ) {
							case 0: calender += " ui-datepicker-hc-group-first";
								cornerClass = " ui-corner-" + ( isRTL ? "right" : "left" ); break;
							case numMonths[ 1 ] - 1: calender += " ui-datepicker-hc-group-last";
								cornerClass = " ui-corner-" + ( isRTL ? "left" : "right" ); break;
							default: calender += " ui-datepicker-hc-group-middle"; cornerClass = ""; break;
						}
					}
					calender += "'>";
				}
				calender += "<div class='ui-datepicker-hc-header ui-widget-header ui-helper-clearfix" + cornerClass + "'>" +
					( /all|left/.test( cornerClass ) && row === 0 ? ( isRTL ? next : prev ) : "" ) +
					( /all|right/.test( cornerClass ) && row === 0 ? ( isRTL ? prev : next ) : "" ) +
					this._generateMonthYearHeader( inst, drawMonth, drawYear, minDate, maxDate,
					row > 0 || col > 0, monthNames, monthNamesShort ) + // draw month headers
					"</div><table class='ui-datepicker-hc-calendar'><thead>" +
					"<tr>";
				thead = ( showWeek ? "<th class='ui-datepicker-hc-week-col'>" + this._get( inst, "weekHeader" ) + "</th>" : "" );
				for ( dow = 0; dow < 7; dow++ ) { // days of the week
					day = ( dow + firstDay ) % 7;
					thead += "<th scope='col'" + ( ( dow + firstDay + 6 ) % 7 >= 5 ? " class='ui-datepicker-hc-week-end'" : "" ) + ">" +
						"<span title='" + dayNames[ day ] + "'>" + dayNamesMin[ day ] + "</span></th>";
				}
				calender += thead + "</tr></thead><tbody>";
				daysInMonth = this._getDaysInMonth( drawYear, drawMonth );
				if ( drawYear === inst.selectedYear && drawMonth === inst.selectedMonth ) {
					inst.selectedDay = Math.min( inst.selectedDay, daysInMonth );
				}
				//hc-ui: 달력 그룹 컨테이너에 인자 추가를 위해 위로 이동 leadDays = ( this._getFirstDayOfMonth( drawYear, drawMonth ) - firstDay + 7 ) % 7;
				curRows = Math.ceil( ( leadDays + daysInMonth ) / 7 ); // calculate the number of rows to generate
				//hc-ui: 멀티플 캘린더일 경우 달 간의 row 갯수 맞추는 로직 삭제; numRows = ( isMultiMonth ? this.maxRows > curRows ? this.maxRows : curRows : curRows ); //If multiple months, use the higher number of rows (see #7043)
				numRows = curRows;
				this.maxRows = numRows;

				//console.log(this._defaults.selectedDateLength)
				//hc-ui: 달력 그룹 컨테이너에 인자 추가를 위해 위로 이동 printDate = this._daylightSavingAdjust( new Date( drawYear, drawMonth, 1 - leadDays ) );
				for ( dRow = 0; dRow < numRows; dRow++ ) { // create date picker rows
					calender += "<tr>";
					tbody = ( !showWeek ? "" : "<td class='ui-datepicker-hc-week-col'>" +
					this._get( inst, "calculateWeek" )( printDate ) + "</td>" );
					for ( dow = 0; dow < 7; dow++ ) { // create date picker days
						daySettings = ( beforeShowDay ?
							beforeShowDay.apply( ( inst.input ? inst.input[ 0 ] : null ), [ printDate ] ) : [ true, "" ] );
							otherMonth = ( printDate.getMonth() !== drawMonth );
							//hc-ui : minSelectableDate 케이스 추가
						unselectable = ( otherMonth && !selectOtherMonths ) || !daySettings[ 0 ] ||
						//( (inst.settings.rangeOneCal && this._defaults.selectedDateLength == 1) && printDate.getTime() < currentDate.getTime()) || //hc-ui : hc-ui : 1캘린더 범위지정 케이스이고 셀렉트된 값이 1개일때 선택날짜 이전 disable  처리
						( (inst.settings.rangeOneCal && $("#" + inst.id).attr("data-selected-length") * 1 == 1) && printDate.getTime() < currentDate.getTime()) || //hc-ui : hc-ui : 1캘린더 범위지정 케이스이고 셀렉트된 값이 1개일때 선택날짜 이전 disable  처리
						( minSelectableDate && printDate < minSelectableDate ) || //hc-ui : range select시 minDate 대체
							( minDate && printDate < minDate ) || 
							( maxSelectableDate && printDate > maxSelectableDate ) || //hc-ui : range select시 maxDate 대체
							( maxDate && printDate > maxDate );
						if(inst.settings.rangeOneCal && $("#" + inst.id).attr("data-selected-length") * 1 == 1){ //hc-ui : 1캘린더 범위지정 케이스이고 셀렉트된 값이 1개일때
							tdHoverable = (printDate.getTime() > currentDate.getTime() && printDate.getTime() <= maxDate) ||
							(printDate.getTime() > currentDate.getTime()); //여기 추가해야함
						}
						
						tbody += "<td class='" +
							( !unselectable && tdHoverable ? this._hoverableClass : "" ) + //hc-ui : tdHoverable 클래스 추가
							( ( dow + firstDay + 6 ) % 7 >= 5 ? " ui-datepicker-hc-week-end" : "" ) + // highlight weekends
							( otherMonth ? " ui-datepicker-hc-other-month" : "" ) + // highlight days from other months
							( ( printDate.getTime() === selectedDate.getTime() && drawMonth === inst.selectedMonth && inst._keyEvent ) || // user pressed key
							( defaultDate.getTime() === printDate.getTime() && defaultDate.getTime() === selectedDate.getTime() ) ?
							// or defaultDate is current printedDate and defaultDate is selectedDate
							" " + this._dayOverClass : "" ) + // highlight selected day
							( unselectable ? " " + this._unselectableClass + " ui-state-disabled" : "" ) +  // highlight unselectable days
							///hc-ui : 아래 3라인, 1캘린더 range 선택시 2개 값 모두 선택되었을 때 range 표현 처리
							( (inst.settings.rangeOneCal && $("#" + inst.id).attr("data-selected-length") * 1 == 2) && (printDate.getTime() > currentDate.getTime() && printDate.getTime() < currentDateTo.getTime() && !otherMonth) ? " ui-datepicker-hc-range" : "" ) + 
							( (inst.settings.rangeOneCal && $("#" + inst.id).attr("data-selected-length") * 1 == 2) && (printDate.getTime() === currentDate.getTime()) ? " ui-datepicker-hc-range-start" : "" ) + 
							( (inst.settings.rangeOneCal && $("#" + inst.id).attr("data-selected-length") * 1 == 2) && (printDate.getTime() === currentDateTo.getTime()) ? " ui-datepicker-hc-range-end" : "" ) + 
							( otherMonth && !showOtherMonths ? "" : " " + daySettings[ 1 ] + // highlight custom dates
							( printDate.getTime() === currentDate.getTime() ? " " + this._currentClass : "" ) + // highlight selected day
							( printDate.getTime() === today.getTime() ? " ui-datepicker-hc-today" : "" ) ) + "'" + // highlight today (if different)
							( ( !otherMonth || showOtherMonths ) && daySettings[ 2 ] ? " title='" + daySettings[ 2 ].replace( /'/g, "&#39;" ) + "'" : "" ) + // cell title
							( unselectable ? "" : " data-handler='selectDay' data-event='click' data-month='" + printDate.getMonth() + "' data-year='" + printDate.getFullYear() + "'" ) + ">" + // actions
							( otherMonth && !showOtherMonths ? "&#xa0;" : // display for other months
							( unselectable ? "<span class='ui-state-default'>" + printDate.getDate() + "</span>" : "<a class='ui-state-default" +
							( printDate.getTime() === today.getTime() ? " ui-state-highlight" : "" ) +
							// hc-ui : backup ( printDate.getTime() === currentDate.getTime() ? " ui-state-active" : "" ) + // highlight selected day
							( ( printDate.getTime() === currentDate.getTime() ) ||  ( currentDateTo && (printDate.getTime() === currentDateTo.getTime() ) ) ? " ui-state-active" : "" ) + //hc-ui: 선택 날짜 하이라이트 처리에 1캘린더 range 선택 시 두번째 날짜 하이라이트 추가
							( otherMonth ? " ui-priority-secondary" : "" ) + // distinguish dates from other months
							"' href='#' aria-current='" + ( printDate.getTime() === currentDate.getTime() ? "true" : "false" ) + // mark date as selected for screen reader
							"' data-date='" + printDate.getDate() + // store date as data
							"'>" + printDate.getDate() + "</a>" ) ) + "</td>"; // display selectable date
						printDate.setDate( printDate.getDate() + 1 );
						printDate = this._daylightSavingAdjust( printDate );
					}
					calender += tbody + "</tr>";
				}
				drawMonth++;
				if ( drawMonth > 11 ) {
					drawMonth = 0;
					drawYear++;
				}
				calender += "</tbody></table>" + ( isMultiMonth ? "</div>" +
							( ( numMonths[ 0 ] > 0 && col === numMonths[ 1 ] - 1 ) ? "<div class='ui-datepicker-hc-row-break'></div>" : "" ) : "" );
				group += calender;
			}
			html += group;
		}
		//hc-ui:스크롤 처리 위해 엘리먼트 3겹 및 스크롤바, 고정 영역 추가
		//console.log(inst.currentDay)
		html += "</div>";
		html += "</div>";
		html += "<div class='ui-datepicker-hc-scrollbar-track'><div class='ui-datepicker-hc-scrollbar'></div></div>";
		html += "<div class='ui-datepicker-hc-control'>";
		//html += "	<strong class='ui-datepicker-hc-current'>" + inst.selectedYear + ". " + (inst.selectedMonth + 1) + "</strong>";
		html += "	<strong class='ui-datepicker-hc-current' data-current-year='0' data-current-month='0'></strong>";
		html += "	<div class='ui-datepicker-hc-btns'>";
		html += "		<span class='ui-datepicker-hc-today'>" + currentText + "</span>";
		html += "		<span class='ui-datepicker-hc-prev'><span class='blind'>" + prevText + "</span></span>";
		html += "		<span class='ui-datepicker-hc-next'><span class='blind'>" + nextText + "</span></span>";
		html += "	</div>";
		html += "</div>";
		html += "</div>";
		html += buttonPanel;
		inst._keyEvent = false;
		return html;
	},

	/* Generate the month and year header. */
	_generateMonthYearHeader: function( inst, drawMonth, drawYear, minDate, maxDate,
			secondary, monthNames, monthNamesShort ) {

		var inMinYear, inMaxYear, month, years, thisYear, determineYear, year, endYear,
			changeMonth = this._get( inst, "changeMonth" ),
			changeYear = this._get( inst, "changeYear" ),
			showMonthAfterYear = this._get( inst, "showMonthAfterYear" ),
			selectMonthLabel = this._get( inst, "selectMonthLabel" ),
			selectYearLabel = this._get( inst, "selectYearLabel" ),
			html = "<div class='ui-datepicker-hc-title'>",
			monthHtml = "";

		// Month selection
		if ( secondary || !changeMonth ) {
			monthHtml += "<span class='ui-datepicker-hc-month'>" + monthNames[ drawMonth ] + "</span>";
		} else {
			inMinYear = ( minDate && minDate.getFullYear() === drawYear );
			inMaxYear = ( maxDate && maxDate.getFullYear() === drawYear );
			monthHtml += "<select class='ui-datepicker-hc-month' aria-label='" + selectMonthLabel + "' data-handler='selectMonth' data-event='change'>";
			for ( month = 0; month < 12; month++ ) {
				if ( ( !inMinYear || month >= minDate.getMonth() ) && ( !inMaxYear || month <= maxDate.getMonth() ) ) {
					monthHtml += "<option value='" + month + "'" +
						( month === drawMonth ? " selected='selected'" : "" ) +
						">" + monthNamesShort[ month ] + "</option>";
				}
			}
			monthHtml += "</select>";
		}

		if ( !showMonthAfterYear ) {
			html += monthHtml + ( secondary || !( changeMonth && changeYear ) ? "&#xa0;" : "" );
		}

		// Year selection
		if ( !inst.yearshtml ) {
			inst.yearshtml = "";
			if ( secondary || !changeYear ) {
				html += "<span class='ui-datepicker-hc-year'>" + drawYear + "</span>";
			} else {

				// determine range of years to display
				years = this._get( inst, "yearRange" ).split( ":" );
				thisYear = new Date().getFullYear();
				determineYear = function( value ) {
					var year = ( value.match( /c[+\-].*/ ) ? drawYear + parseInt( value.substring( 1 ), 10 ) :
						( value.match( /[+\-].*/ ) ? thisYear + parseInt( value, 10 ) :
						parseInt( value, 10 ) ) );
					return ( isNaN( year ) ? thisYear : year );
				};
				year = determineYear( years[ 0 ] );
				endYear = Math.max( year, determineYear( years[ 1 ] || "" ) );
				year = ( minDate ? Math.max( year, minDate.getFullYear() ) : year );
				endYear = ( maxDate ? Math.min( endYear, maxDate.getFullYear() ) : endYear );
				inst.yearshtml += "<select class='ui-datepicker-hc-year' aria-label='" + selectYearLabel + "' data-handler='selectYear' data-event='change'>";
				for ( ; year <= endYear; year++ ) {
					inst.yearshtml += "<option value='" + year + "'" +
						( year === drawYear ? " selected='selected'" : "" ) +
						">" + year + "</option>";
				}
				inst.yearshtml += "</select>";

				html += inst.yearshtml;
				inst.yearshtml = null;
			}
		}

		html += this._get( inst, "yearSuffix" );
		if ( showMonthAfterYear ) {
			html += ( secondary || !( changeMonth && changeYear ) ? "&#xa0;" : "" ) + monthHtml;
		}
		html += "</div>"; // Close datepickerHC_header
		return html;
	},

	/* Adjust one of the date sub-fields. */
	_adjustInstDate: function( inst, offset, period ) {
		var year = inst.selectedYear + ( period === "Y" ? offset : 0 ),
			month = inst.selectedMonth + ( period === "M" ? offset : 0 ),
			day = Math.min( inst.selectedDay, this._getDaysInMonth( year, month ) ) + ( period === "D" ? offset : 0 ),
			date = this._restrictMinMax( inst, this._daylightSavingAdjust( new Date( year, month, day ) ) );

		inst.selectedDay = date.getDate();
		inst.drawMonth = inst.selectedMonth = date.getMonth();
		inst.drawYear = inst.selectedYear = date.getFullYear();
		if ( period === "M" || period === "Y" ) {
			this._notifyChange( inst );
		}
	},

	/* Ensure a date is within any min/max bounds. */
	_restrictMinMax: function( inst, date ) {
		var minDate = this._getMinMaxDate( inst, "min" ),
			maxDate = this._getMinMaxDate( inst, "max" ),
			newDate = ( minDate && date < minDate ? minDate : date );
		return ( maxDate && newDate > maxDate ? maxDate : newDate );
	},

	/* Notify change of month/year. */
	_notifyChange: function( inst ) {
		var onChange = this._get( inst, "onChangeMonthYear" );
		if ( onChange ) {
			onChange.apply( ( inst.input ? inst.input[ 0 ] : null ),
				[ inst.selectedYear, inst.selectedMonth + 1, inst ] );
		}
	},

	/* Determine the number of months to show. */
	_getNumberOfMonths: function( inst ) {
		var numMonths = this._get( inst, "numberOfMonths" );
		return ( numMonths == null ? [ 1, 1 ] : ( typeof numMonths === "number" ? [ 1, numMonths ] : numMonths ) );
	},

	/* Determine the current maximum date - ensure no time components are set. */
	_getMinMaxDate: function( inst, minMax ) {
		return this._determineDate( inst, this._get( inst, minMax + "Date" ), null );
	},

	/* Find the number of days in a given month. */
	_getDaysInMonth: function( year, month ) {
		return 32 - this._daylightSavingAdjust( new Date( year, month, 32 ) ).getDate();
	},

	/* Find the day of the week of the first of a month. */
	_getFirstDayOfMonth: function( year, month ) {
		return new Date( year, month, 1 ).getDay();
	},

	/* Determines if we should allow a "next/prev" month display change. */
	_canAdjustMonth: function( inst, offset, curYear, curMonth ) {
		var numMonths = this._getNumberOfMonths( inst ),
			date = this._daylightSavingAdjust( new Date( curYear,
			curMonth + ( offset < 0 ? offset : numMonths[ 0 ] * numMonths[ 1 ] ), 1 ) );

		if ( offset < 0 ) {
			date.setDate( this._getDaysInMonth( date.getFullYear(), date.getMonth() ) );
		}
		return this._isInRange( inst, date );
	},

	/* Is the given date in the accepted range?  hc-ui : 레인지 셀렉터 기간 하이라이트 시 참조할 수 있는지 확인 */
	_isInRange: function( inst, date ) {
		//console.log("_isInRange")
		var yearSplit, currentYear,
			minDate = this._getMinMaxDate( inst, "min" ),
			maxDate = this._getMinMaxDate( inst, "max" ),
			minYear = null,
			maxYear = null,
			years = this._get( inst, "yearRange" );
			if ( years ) {
				yearSplit = years.split( ":" );
				currentYear = new Date().getFullYear();
				minYear = parseInt( yearSplit[ 0 ], 10 );
				maxYear = parseInt( yearSplit[ 1 ], 10 );
				if ( yearSplit[ 0 ].match( /[+\-].*/ ) ) {
					minYear += currentYear;
				}
				if ( yearSplit[ 1 ].match( /[+\-].*/ ) ) {
					maxYear += currentYear;
				}
			}

		return ( ( !minDate || date.getTime() >= minDate.getTime() ) &&
			( !maxDate || date.getTime() <= maxDate.getTime() ) &&
			( !minYear || date.getFullYear() >= minYear ) &&
			( !maxYear || date.getFullYear() <= maxYear ) );
	},

	/* Provide the configuration settings for formatting/parsing. */
	_getFormatConfig: function( inst ) {
		var shortYearCutoff = this._get( inst, "shortYearCutoff" );
		shortYearCutoff = ( typeof shortYearCutoff !== "string" ? shortYearCutoff :
			new Date().getFullYear() % 100 + parseInt( shortYearCutoff, 10 ) );
		return { shortYearCutoff: shortYearCutoff,
			dayNamesShort: this._get( inst, "dayNamesShort" ), dayNames: this._get( inst, "dayNames" ),
			monthNamesShort: this._get( inst, "monthNamesShort" ), monthNames: this._get( inst, "monthNames" ) };
	},

	/* Format the given date for display. */
	_formatDate: function( inst, day, month, year ) {
		if ( !day ) {
			inst.currentDay = inst.selectedDay;
			inst.currentMonth = inst.selectedMonth;
			inst.currentYear = inst.selectedYear;
		}
		var date = ( day ? ( typeof day === "object" ? day :
			this._daylightSavingAdjust( new Date( year, month, day ) ) ) :
			this._daylightSavingAdjust( new Date( inst.currentYear, inst.currentMonth, inst.currentDay ) ) );
		return this.formatDate( this._get( inst, "dateFormat" ), date, this._getFormatConfig( inst ) );
	}
} );

/*
 * Bind hover events for datepickerHC elements.
 * Done via delegate so the binding only occurs once in the lifetime of the parent div.
 * Global datepickerHC_instActive, set by _updateDatepickerHC allows the handlers to find their way back to the active picker.
 */
function datepickerHC_bindHover( dpDiv ) {
	var selector = "button, .ui-datepicker-hc-prev, .ui-datepicker-hc-next, .ui-datepicker-hc-calendar td a";
	return dpDiv.on( "mouseout", selector, function() {
			$( this ).removeClass( "ui-state-hover" );
			if ( this.className.indexOf( "ui-datepicker-hc-prev" ) !== -1 ) {
				$( this ).removeClass( "ui-datepicker-hc-prev-hover" );
			}
			if ( this.className.indexOf( "ui-datepicker-hc-next" ) !== -1 ) {
				$( this ).removeClass( "ui-datepicker-hc-next-hover" );
			}
		} )
		.on( "mouseover", selector, datepickerHC_handleMouseover );
}

function datepickerHC_handleMouseover() {
	/*
	console.log("datepickerHC_handleMouseover")
	var cur = $(".ui-datepicker-hc-current-day")
	console.log($(this))
	*/
	//console.log("_handleMouseover : " + this._defaults.selectedDateLength)
	if ( !$.datepickerHC._isDisabledDatepickerHC( datepickerHC_instActive.inline ? datepickerHC_instActive.dpDiv.parent()[ 0 ] : datepickerHC_instActive.input[ 0 ] ) ) {
		$( this ).parents( ".ui-datepicker-hc-calendar" ).find( "a" ).removeClass( "ui-state-hover" );
		$( this ).addClass( "ui-state-hover" );
		if ( this.className.indexOf( "ui-datepicker-hc-prev" ) !== -1 ) {
			$( this ).addClass( "ui-datepicker-hc-prev-hover" );
		}
		if ( this.className.indexOf( "ui-datepicker-hc-next" ) !== -1 ) {
			$( this ).addClass( "ui-datepicker-hc-next-hover" );
		}
	}
}

/* jQuery extend now ignores nulls! */
function datepickerHC_extendRemove( target, props ) {
	$.extend( target, props );
	for ( var name in props ) {
		if ( props[ name ] == null ) {
			target[ name ] = props[ name ];
		}
	}
	return target;
}

/* Invoke the datepickerHC functionality.
   @param  options  string - a command, optionally followed by additional parameters or
					Object - settings for attaching new datepickerHC functionality
   @return  jQuery object */
$.fn.datepickerHC = function( options ) {

	/* Verify an empty collection wasn't passed - Fixes #6976 */
	if ( !this.length ) {
		return this;
	}

	/* Initialise the date picker. */
	if ( !$.datepickerHC.initialized ) {
		//$( document ).on( "mousedown", $.datepickerHC._checkExternalClick );
		$( document ).on( "mousedown", $.datepickerHC._checkExternalClick );
		$.datepickerHC.initialized = true;
	}

	/* Append datepickerHC main container to body if not exist. */
	if ( $( "#" + $.datepickerHC._mainDivId ).length === 0 ) {
		$( "body" ).append( $.datepickerHC.dpDiv );
	}

	var otherArgs = Array.prototype.slice.call( arguments, 1 );
	if ( typeof options === "string" && ( options === "isDisabled" || options === "getDate" || options === "widget" ) ) {
		return $.datepickerHC[ "_" + options + "DatepickerHC" ].
			apply( $.datepickerHC, [ this[ 0 ] ].concat( otherArgs ) );
	}
	if ( options === "option" && arguments.length === 2 && typeof arguments[ 1 ] === "string" ) {
		return $.datepickerHC[ "_" + options + "DatepickerHC" ].
			apply( $.datepickerHC, [ this[ 0 ] ].concat( otherArgs ) );
	}
	return this.each( function() {
		if ( typeof options === "string" ) {
			$.datepickerHC[ "_" + options + "DatepickerHC" ]
				.apply( $.datepickerHC, [ this ].concat( otherArgs ) );
		} else {
			$.datepickerHC._attachDatepickerHC( this, options );
		}

		$(document).on('keyup', '.hasDatepickerHC', function(){
			var  $keyThis = $(this);
			if($keyThis.val().length > 0 || !$keyThis.val() == ''){
				$keyThis.closest('label').addClass('on');
			}else{
				$keyThis.closest('label').removeClass('on');
			}
		});

		$('.hasDatepickerHC').on('focusin', function(){
			$(this).closest('label').addClass('focused');
		}).on('focusout', function(){
			$(this).closest('label').removeClass('focused');
		})
	} );

};

$.datepickerHC = new DatepickerHC(); // singleton instance
$.datepickerHC.initialized = false;
$.datepickerHC.uuid = new Date().getTime();
$.datepickerHC.version = "@VERSION";

return $.datepickerHC;

} );
