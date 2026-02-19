*"----------------------------------------------------------------------
*"* Sample ABAP for testing
*"----------------------------------------------------------------------
CLASS zcl_sample DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS process.
  PRIVATE SECTION.
    DATA mv_value TYPE string.
ENDCLASS.

CLASS zcl_sample IMPLEMENTATION.
  METHOD process.
    DATA lv_a TYPE string.
    DATA lv_b TYPE string.
    lv_a = 'hello'.
    mv_value = mv_value + 1.
    mv_value = |{ lv_a }{ lv_b }|.
IF lv_a IS NOT INITIAL.
  RETURN.
ENDIF.
    me->process( ).
  ENDMETHOD.
ENDCLASS.
